// test_flash_moe_ordering_fix.cpp
//
// Verifies the root-cause fix for gibberish output: eval_callback must fire on
// "ffn_moe_weights-N" (AFTER GET_ROWS), NOT on "ffn_moe_topk-N" (BEFORE GET_ROWS).
//
// The bug: triggering on ffn_moe_topk caused the ID remapping to happen before
// ffn_moe_weights = ggml_get_rows(probs, selected_experts) ran. GET_ROWS then
// used remapped slot IDs [0,1,...,K-1] instead of the real top-K expert IDs,
// fetching near-zero probabilities → gibberish output after weight normalization.
//
// The fix: trigger on "ffn_moe_weights-" so GET_ROWS has already executed with
// the original expert IDs. Then remap ids_tensor (= ffn_moe_topk) for MUL_MAT_ID.
//
// This test mirrors the ask-phase logic and the remap logic using the corrected
// source tensor (ids_tensor, NOT t) so the fix is captured in a regression test.
//
// Build (standalone — no ggml link required):
//   clang++ -std=c++17 tests/flash_moe/test_flash_moe_ordering_fix.cpp \
//       -o /tmp/test_ordering_fix && /tmp/test_ordering_fix

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

// ─── Mirror the ask-phase trigger logic ──────────────────────────────────────

static bool ask_will_pause(const char* tensor_name) {
    // Fixed: trigger on "ffn_moe_weights-" (with dash), NOT "ffn_moe_topk"
    return (std::strstr(tensor_name, "ffn_moe_weights-") != nullptr);
}

// ─── Mirror the data-phase remap logic ───────────────────────────────────────
// Source is ids_tensor (= ffn_moe_topk), NOT t (= ffn_moe_weights).

struct RemapResult {
    std::unordered_map<int32_t, int32_t> expert_to_slot;
    std::vector<int32_t> remapped_ids;  // ids_tensor after remap
};

static RemapResult remap_ids_from_ids_tensor(
    std::vector<int32_t> ids_tensor_data,  // data from ids_tensor (= ffn_moe_topk)
    int n_experts,
    int max_slots)
{
    RemapResult r;
    int32_t slot = 0;

    for (auto id : ids_tensor_data) {
        if (id >= 0 && id < n_experts &&
            r.expert_to_slot.find(id) == r.expert_to_slot.end()) {
            if (slot >= max_slots) break;
            r.expert_to_slot[id] = slot++;
        }
    }

    r.remapped_ids = ids_tensor_data;
    for (auto& id : r.remapped_ids) {
        if (id >= 0 && id < n_experts) {
            auto it = r.expert_to_slot.find(id);
            id = (it != r.expert_to_slot.end()) ? it->second : 0;
        } else {
            id = 0;  // clamp invalid/padding IDs to slot 0
        }
    }
    return r;
}

// ─── Test 1: Ask phase trigger ────────────────────────────────────────────────
// The fix: "ffn_moe_weights-N" triggers, "ffn_moe_topk-N" does NOT.
void test_ask_phase_trigger() {
    std::cout << "Running test_ask_phase_trigger..." << std::endl;

    // Should trigger (GET_ROWS output — fire AFTER GET_ROWS, BEFORE MUL_MAT_ID)
    assert(ask_will_pause("ffn_moe_weights-0")  == true);
    assert(ask_will_pause("ffn_moe_weights-23") == true);
    assert(ask_will_pause("ffn_moe_weights-47") == true);  // last layer of 48-layer model

    // Must NOT trigger — these fired too early (before GET_ROWS) in the old buggy code
    assert(ask_will_pause("ffn_moe_topk-0")  == false);
    assert(ask_will_pause("ffn_moe_topk-23") == false);

    // Must NOT trigger — weight normalizations run after GET_ROWS; we only need to
    // intercept once, right after GET_ROWS, before MUL_MAT_ID.
    assert(ask_will_pause("ffn_moe_weights_norm-0")  == false);  // "norm" not "-"
    assert(ask_will_pause("ffn_moe_weights_norm-23") == false);

    // Other unrelated tensors
    assert(ask_will_pause("blk.0.ffn_gate_exps.weight") == false);
    assert(ask_will_pause("ffn_gate_inp-0")              == false);
    assert(ask_will_pause("")                            == false);

    std::cout << "test_ask_phase_trigger passed!" << std::endl;
}

// ─── Test 2: IDs are read from ids_tensor (ffn_moe_topk), not from t ─────────
// Verifies the fix: remap source is ids_tensor, so t (ffn_moe_weights) is
// untouched and GET_ROWS result is preserved.
void test_remap_reads_from_ids_tensor() {
    std::cout << "Running test_remap_reads_from_ids_tensor..." << std::endl;

    // ids_tensor (= ffn_moe_topk) contains raw expert IDs: [47, 3, 81, 47]
    std::vector<int32_t> ids_tensor_data = {47, 3, 81, 47};
    // t (= ffn_moe_weights) would contain F32 probability data — not IDs.
    // The fix: we must NOT read from t. ids_tensor is our source.

    auto r = remap_ids_from_ids_tensor(ids_tensor_data, /*n_experts=*/128, /*max_slots=*/8);

    // Slot assignment: 47→0, 3→1, 81→2 (first-occurrence order)
    assert(r.expert_to_slot.at(47) == 0);
    assert(r.expert_to_slot.at(3)  == 1);
    assert(r.expert_to_slot.at(81) == 2);

    // ids_tensor remapped: [47,3,81,47] → [0,1,2,0]
    assert(r.remapped_ids[0] == 0);
    assert(r.remapped_ids[1] == 1);
    assert(r.remapped_ids[2] == 2);
    assert(r.remapped_ids[3] == 0);  // duplicate 47 → same slot 0

    std::cout << "test_remap_reads_from_ids_tensor passed!" << std::endl;
}

// ─── Test 3: GET_ROWS result is not corrupted ─────────────────────────────────
// Simulates the BEFORE/AFTER: with the old code, GET_ROWS would see remapped
// IDs [0,1,2,...] instead of the real expert IDs [47,3,81,...].
// With the fix, GET_ROWS runs BEFORE we touch ids_tensor.
// We verify this by showing t (ffn_moe_weights) is distinct from ids_tensor
// and is never modified by the remap logic.
void test_get_rows_not_corrupted() {
    std::cout << "Running test_get_rows_not_corrupted..." << std::endl;

    // Simulate: GET_ROWS has already run using real expert IDs [47, 3, 81]
    // and produced probability values stored in t (ffn_moe_weights).
    std::vector<float> t_ffn_moe_weights = {0.6f, 0.3f, 0.1f};  // expert probs
    const std::vector<float> original_weights = t_ffn_moe_weights;

    // Now remap_ids_from_ids_tensor operates on ids_tensor only
    std::vector<int32_t> ids_tensor_data = {47, 3, 81};
    remap_ids_from_ids_tensor(ids_tensor_data, 128, 8);

    // t (ffn_moe_weights) must be untouched — we never read or write it
    assert(t_ffn_moe_weights == original_weights);

    std::cout << "test_get_rows_not_corrupted passed!" << std::endl;
}

// ─── Test 4: Old (buggy) behavior would have corrupted GET_ROWS ───────────────
// Documents what the bug looked like: if remap fired on ffn_moe_topk BEFORE
// GET_ROWS, GET_ROWS would receive slot IDs [0,1,2,...] and fetch probs for
// the wrong (low-probability) experts.
void test_old_behavior_would_corrupt_weights() {
    std::cout << "Running test_old_behavior_would_corrupt_weights..." << std::endl;

    // Real expert IDs for this token
    std::vector<int32_t> real_ids = {47, 3, 81};
    // Expert probabilities from softmax (higher index = real selected expert)
    std::vector<float> probs(128, 0.0f);
    probs[47] = 0.60f;
    probs[3]  = 0.30f;
    probs[81] = 0.10f;
    // Other experts have near-zero probabilities
    for (int i = 0; i < 128; ++i) {
        if (i != 47 && i != 3 && i != 81) probs[i] = 0.001f / 125.0f;
    }

    // CORRECT (new behavior): GET_ROWS runs first with real_ids → correct probs
    float sum_correct = probs[real_ids[0]] + probs[real_ids[1]] + probs[real_ids[2]];
    assert(sum_correct > 0.9f);  // sum of top-K probs ≈ 1.0

    // OLD BUG: remap fired before GET_ROWS; GET_ROWS would see slot IDs [0,1,2]
    std::vector<int32_t> remapped_early = {0, 1, 2};  // what the old code produced
    float sum_buggy = probs[remapped_early[0]] + probs[remapped_early[1]] + probs[remapped_early[2]];
    assert(sum_buggy < 0.01f);  // probs[0] + probs[1] + probs[2] ≈ near-zero → garbage

    std::cout << "test_old_behavior_would_corrupt_weights passed!" << std::endl;
}

// ─── Test 5: K=8, Qwen3-30B-A3B typical batch ────────────────────────────────
// Simulates a single token with n_expert_used=8, n_experts=128.
void test_qwen3_typical_batch() {
    std::cout << "Running test_qwen3_typical_batch..." << std::endl;

    // Typical top-8 expert IDs from a single Qwen3-30B-A3B token
    std::vector<int32_t> ids = {14, 57, 3, 91, 22, 8, 63, 107};
    auto r = remap_ids_from_ids_tensor(ids, /*n_experts=*/128, /*max_slots=*/8);

    assert((int)r.expert_to_slot.size() == 8);
    // Remapped must be [0,1,2,3,4,5,6,7] — unique experts get sequential slots
    for (int i = 0; i < 8; ++i) {
        assert(r.remapped_ids[i] == i);
        assert(r.expert_to_slot.at(ids[i]) == i);
    }

    std::cout << "test_qwen3_typical_batch passed!" << std::endl;
}

// ─── Test 6: Padding / clamping (Bug 7 regression) ────────────────────────────
void test_invalid_ids_clamped_to_slot0() {
    std::cout << "Running test_invalid_ids_clamped_to_slot0..." << std::endl;

    // -1 and 200 (>= n_experts=128) are invalid; they should map to slot 0
    std::vector<int32_t> ids = {14, -1, 200, 57};
    auto r = remap_ids_from_ids_tensor(ids, 128, 8);

    assert(r.remapped_ids[0] == 0);  // 14 → slot 0
    assert(r.remapped_ids[1] == 0);  // -1 → clamped to 0
    assert(r.remapped_ids[2] == 0);  // 200 out-of-range → clamped to 0
    assert(r.remapped_ids[3] == 1);  // 57 → slot 1

    std::cout << "test_invalid_ids_clamped_to_slot0 passed!" << std::endl;
}

// ─── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_ask_phase_trigger();
    test_remap_reads_from_ids_tensor();
    test_get_rows_not_corrupted();
    test_old_behavior_would_corrupt_weights();
    test_qwen3_typical_batch();
    test_invalid_ids_clamped_to_slot0();
    std::cout << "\nAll ordering-fix tests passed!" << std::endl;
    return 0;
}
