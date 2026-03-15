# Feature: Prompt Lookup Decoding (ngram-based speculative decoding)

## What It Does

`--promptlookup N` enables hash-based self-speculative decoding without a draft model. The growing context is indexed in an O(1) hash table that maps n-grams (sequences of N tokens) to their predicted successor. When the main model is about to evaluate a single token, the hash is queried first; if a prediction chain is found, a batch of drafted tokens is evaluated in one shot and verified using the existing speculative decoding loop.

**No draft model needed. No extra GPU memory beyond the hash table.**

Flag: `--promptlookup N` (integer, 0=disabled). Shared with draft model: `--draftamount` controls how many tokens to draft per chunk. Mutually exclusive with `--draftmodel` (draft model takes priority).

---

## Implementation Strategy

### Phase 1: Linear backward scan (abandoned)

The first attempt used a simple backward scan: search `context_tokens` from the end for the last occurrence of the final `ngram_min` tokens, then copy the continuation. This failed completely — with `ngram_min=2`, the pair `\n[` matched before `[LOG...]`, `[INFO...]`, and `[SISTEMA...]` equally, always producing the wrong continuation. First-token match rate was ~0%, so no speculative benefit.

### Phase 2: Hash-based approach — `common_ngram_mod`

Switched to upstream llama.cpp PR #19164's `common_ngram_mod` (already present in `common/ngram-mod.h`). This is an O(1) open-addressed hash table:

- `add(tokens)` — stores `tokens[n]` as the prediction for key `tokens[0..n-1]`
- `get(tokens)` — returns predicted next token or `EMPTY (-1)` on miss
- Hash: `res = res * 6364136223846793005ULL + tokens[i]` mod table_size
- Last-write-wins on collision (no chaining)
- Occupancy reset: when `used/size > 0.25`, table is cleared and repopulated from scratch to restore quality

With `n=16`, a 16-token key uniquely identifies a specific position in nearly any repetitive context, avoiding the false-positive storm that broke Phase 1.

### Critical bug found during Phase 2: off-by-one in key construction

KoboldCpp's `current_context_tokens` already contains `embd[0]` as its **last element** (it is appended at line ~4891, before `embd` is populated). The initial key build had an off-by-one:

```cpp
// BUGGY: ctx_size - n + 1 + i
// When i = n-2: result[n-2] = context_tokens[ctx_size-1] = embd[0]
// Then:         result[n-1] = embd[0]    ← embd[0] duplicated!
for (size_t i = 0; i < n - 1; ++i) {
    result[i] = context_tokens[ctx_size - n + 1 + i];  // BUG
}
result[n - 1] = embd[0];
```

This placed `embd[0]` at both positions `n-2` and `n-1` of the key. No such double-repeated n-gram ever appears in context, so `get()` always returned `EMPTY` → zero drafts attempted → no stats printed → appeared as "no draft tentative at all."

The fix is a one-character change:

```cpp
// FIXED: ctx_size - n + i
// Key = context_tokens[ctx_size-n .. ctx_size-2] + embd[0]  (all distinct)
for (size_t i = 0; i < n - 1; ++i) {
    result[i] = context_tokens[ctx_size - n + i];
}
result[n - 1] = embd[0];
```

This was diagnosed by writing `test_ngram_lookup.py`, a Python simulation of the full algorithm, and adding a regression test `test_key_does_not_duplicate_last_token` that proves the buggy key never hits and the correct key always does.

---

## Performance Findings

After the fix, drafting worked correctly — predictions are generated and verified. However, **real-world performance gain is marginal at best**. The reasons:

1. **Hash collisions limit prediction quality.** Even with `n=16`, real prompts contain enough variety that many predicted tokens are wrong. Each mismatch aborts the draft chunk, wasting the batch evaluation overhead.

2. **No GPU-side benefit on small mismatches.** The speculative gain only materialises when several consecutive tokens are accepted. If the first drafted token mismatches (common in generative text), the batch eval cost is paid with zero benefit.

3. **Works best on highly repetitive, templated text.** Identical repeated lines (e.g., copy-pasted log entries, fill-in-the-blank templates) can achieve high acceptance rates. Ordinary conversation or creative text gains little.

4. **Draft model is strictly better when available.** A draft model predicts based on the probability distribution; ngram lookup predicts based on exact past occurrences. For non-repetitive output, the draft model wins overwhelmingly.

**Recommended use cases:** structured output with repeated patterns, code with boilerplate, fill-in forms, tokenizer stress tests. Not useful for open-ended generation.

---

## Files Modified

| File | Change |
|------|--------|
| `expose.h` | Added `const int lookup_ngram_min = 0;` to `load_model_inputs` |
| `koboldcpp.py` | Added ctypes field, `--promptlookup` argparse arg, wiring in `load_model()` |
| `gpttype_adapter.cpp` | Include, globals, model-load init, generation-reset, new `prompt_lookup_eval_chunk()`, generation loop condition, speculative eval branch |
| `Makefile` | Added `ngram-mod.o` to all four `OBJS_*` lists and build rule |
| `test_ngram_lookup.py` | Python simulation + unit tests for the algorithm (11 tests) |

---

## Architecture

### Globals (`gpttype_adapter.cpp`)

```cpp
#include "common/ngram-mod.h"

int lookup_ngram_min = 0;           // 0=disabled; set from inputs.lookup_ngram_min at load
common_ngram_mod * ngram_mod_state = nullptr;
size_t ngram_mod_i_last = 0;        // last context_size when hash was updated
```

### Model load

```cpp
if (ngram_mod_state != nullptr) { delete ngram_mod_state; ngram_mod_state = nullptr; }
lookup_ngram_min = inputs.lookup_ngram_min;
if (lookup_ngram_min > 0 && draft_ctx == nullptr && file_format == FileFormat::GGUF_GENERIC) {
    // guard: recurrent/hybrid models not supported
    speculative_chunk_amt = inputs.draft_amount;
    ngram_mod_state = new common_ngram_mod(lookup_ngram_min, 4*1024*1024);
    // warn if n < 16
}
```

### Generation start (reset)

```cpp
if (ngram_mod_state != nullptr) {
    ngram_mod_state->reset();
    ngram_mod_i_last = 0;
}
```

### `prompt_lookup_eval_chunk()`

1. Incrementally add new ngrams to hash (only since `ngram_mod_i_last`, with n-token overlap for boundary continuity)
2. Occupancy guard: if `>25%` full, reset and repopulate from scratch
3. Build initial key: `context_tokens[ctx_size-n .. ctx_size-2] + embd[0]`
4. Greedily draft up to `max_draft` tokens via sliding-window hash lookups
5. Batch-eval `[embd[0], drafted[0..N-2]]` through main model
6. Return logits for speculative verification loop

### Generation loop

- Condition: `(draft_ctx==nullptr && lookup_ngram_min<=0)` skips speculative path when neither draft model nor lookup is active
- Branch: `draft_ctx != nullptr` → draft model path; else → prompt lookup path
- Fallback: if lookup returns `draft_success=false` (no ngram match), fall back to normal single-token eval in the same branch — no crash, no wasted work