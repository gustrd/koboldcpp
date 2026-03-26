# Flash-MoE Quality Fix for GPT-OSS-120B (and Similar Models)

## The Problem: Severe Quality Degradation
When running models like **GPT-OSS-120B** with Flash-MoE, you may encounter severe quality degradation (e.g., incoherent outputs or looping logic), despite the generation speed or basic VRAM footprint appearing normal. 

The root cause stems from how Flash-MoE dynamically allocates memory slots for its active expert computations and how it handles missing metadata.

## The Root Cause: Silent Expert "Clamping"

Flash-MoE is designed to aggressively shrink RAM usage by only allocating a small number of memory slots (K-Slots) per token. The issue occurs if your model routes to **more than 8 active experts per token**, but the engine only allocates **8 slots**.

### 1. Metadata Extraction Failure (`extract_experts.py`)
During the extraction phase, the python script reads the model's GGUF KV metadata to identify the `expert_used_count` (e.g., looking for `<arch>.expert_used_count` or `llama.expert_used_count`). 
Because custom architectures or models like `GPT-OSS-120B` often use non-standard naming schemas for this parameter, the python script fails to retrieve it and silently returns `None`. Consequently, the generated `expert_index.json` is missing the `n_expert_used` parameter.

### 2. The Hardcoded Default (K=8)
When KoboldCpp loads the model, `flash_moe_manager.cpp` reads the `expert_index.json`. Without the explicit `n_expert_used` value, the engine employs a safety fallback and assumes **8 active experts**:
```cpp
n_expert_used = index["config"].value("n_expert_used", 8);
```
Flash-MoE then rigorously binds its tensor size to 8 memory slots (`tensor->ne[2] = 8`).

### 3. The "Clamping" Trap
If GPT-OSS-120B actually requests **16 active experts** per token during a forward pass, the Flash-MoE loader `load_and_remap_layer` attempts to map all 16 unique experts into the available memory slots.
- **Experts 1-8:** Correctly loaded into slots 0 through 7.
- **Experts 9-16:** Because there are no remaining slots (and mapping an unallocated slot would cause an immediate Out-Of-Bounds crash), the engine enforces the following safety check:

```cpp
if (next_slot < mgr.n_expert_used) {
    eid_to_slot[id] = next_slot++;
} else {
    eid_to_slot[id] = 0; // Clamp
}
```
**This silently routes half of the active calculations for the token through the weights of the expert sitting in slot `0`.** Repeating the same incorrect computational weights over and over catastrophically damages the model's intelligence, turning the output into garbage.

---

## The Solution: Manually Edit the Index

To fix this immediately, you must bypass the fallback by defining the exact active expert count in the index file.

1. Navigate to your extracted experts directory (e.g., `flash_moe_experts/`).
2. Open `expert_index.json` in a text editor.
3. Locate the `"config"` dictionary at the top of the file.
4. Manually add the `"n_expert_used"` key with the correct integer value for GPT-OSS-120B (e.g., `16`, or however many the model routes).

**Example:**
```json
{
  "config": {
    "n_layers": 48,
    "n_experts": 128,
    "token_id_base": 200000,
    "token_id_count": 6144,
    "n_expert_used": 16  <-- ADD THE EXACT CONFIG HERE
  },
  "experts": {
    ...
  }
}
```

As soon as this value is provided, Flash-MoE will properly expand the K-Slot tensor allocation, bypass the clamping threshold, load all required experts simultaneously into distinct slots, and instantly restore the model's expected generation quality.
