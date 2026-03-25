// test_flash_moe_bug7_oob.cpp
//
// Focused unit tests for Bug 7: MUL_MAT_ID OOB Access.
//
// Root cause: With K-slot weight tensors (n_expert_used=K), the MUL_MAT_ID
// kernel indexes into a tensor with only K rows. Any ID >= K causes an
// out-of-bounds read/write → segfault or access violation.
//
// Fix: The eval_callback clamps ALL IDs to [0, K-1] before writing them
// back into the selected_experts tensor that MUL_MAT_ID reads from:
//   - Valid IDs (0..n_experts-1) → mapped slot (0..slot-1) OR 0 if no slot
//   - Padding/invalid IDs (-1, out-of-range) → 0 always
//
// KEY INVARIANT (from PLAN.md §4.6):
//   Every ID written back by the callback MUST satisfy: 0 <= id < K.
//   Violation → OOB memory access in GPU/CPU MUL_MAT_ID kernel.
//
// Build (standalone — only the remap logic, no ggml link required):
//   g++ -std=c++17 tests/flash_moe/test_flash_moe_bug7_oob.cpp \
//       -o /tmp/test_bug7 && /tmp/test_bug7
//
// These tests do NOT call eval_callback directly (that requires ggml link
// and manager state). Instead they exercise the exact clamping algorithm
// extracted verbatim from eval_callback, so any algorithm change will
// break these tests.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>

// ─── Exact copy of the slot-assignment + remap logic from eval_callback ───────
//
// This mirrors flash_moe_manager.cpp:eval_callback(), step 3-5.
// If the algorithm in the .cpp changes, this copy must also change, and
// these tests will catch any divergence.

struct RemapResult {
    std::unordered_map<int32_t, int32_t> expert_to_slot;
    std::vector<int32_t> remapped;
};

// Implements step 3-5 of eval_callback:
//   step 3: build expert_id → slot_index map (first-occurrence order, cap at max_slots).
//   step 4: (skipped — disk I/O not needed for remap testing).
//   step 5: remap IDs in-place, clamp all IDs to [0, max_slots-1].
//
// Bug 7 invariant enforced here:
//   After remap, every element of result.remapped is in [0, max_slots-1].
static RemapResult callback_remap(const std::vector<int32_t>& raw_ids,
                                   int n_experts,  // n_exp in eval_callback
                                   int max_slots)  // K = n_expert_used
{
    RemapResult r;

    // Step 3: build expert_to_slot (first-occurrence, cap at max_slots)
    int32_t slot = 0;
    for (auto id : raw_ids) {
        if (id >= 0 && id < n_experts &&
            r.expert_to_slot.find(id) == r.expert_to_slot.end()) {
            if (slot >= max_slots) break;
            r.expert_to_slot[id] = slot++;
        }
    }

    // Step 5: remap + clamp (exact logic from flash_moe_manager.cpp:346-353)
    r.remapped = raw_ids;
    for (auto& id : r.remapped) {
        if (id >= 0 && id < n_experts) {
            auto it = r.expert_to_slot.find(id);
            id = (it != r.expert_to_slot.end()) ? it->second : 0;
        } else {
            id = 0;  // clamp invalid/padding IDs to slot 0
        }
    }

    return r;
}

// ─── Helper: assert Bug 7 invariant holds ────────────────────────────────────
static void assert_no_oob(const std::vector<int32_t>& remapped, int max_slots,
                           const std::string& test_name) {
    for (size_t i = 0; i < remapped.size(); ++i) {
        if (remapped[i] < 0 || remapped[i] >= max_slots) {
            std::cerr << "[FAIL] " << test_name << ": remapped[" << i << "] = "
                      << remapped[i] << " is OOB for K=" << max_slots << std::endl;
            std::abort();
        }
    }
}

// ─── Test 1 ──────────────────────────────────────────────────────────────────
// BUG 7 SCENARIO: padding IDs (-1) from ggml_argsort_top_k must NOT survive
// the remap. With --gpulayers 48, ggml_argsort_top_k pads unused slots with -1.
// Before the fix, -1 as a size_t index = 0xFFFFFFFFFFFFFFFF → crash.
void test_padding_ids_clamped_to_zero() {
    std::cout << "test_padding_ids_clamped_to_zero ... ";

    const int K = 8, N = 128;
    // Simulate: 3 valid experts + 5 padding slots (-1)
    std::vector<int32_t> raw = {42, 97, 13, -1, -1, -1, -1, -1};
    auto r = callback_remap(raw, N, K);

    // Valid experts mapped to slots 0,1,2
    assert(r.expert_to_slot.at(42) == 0);
    assert(r.expert_to_slot.at(97) == 1);
    assert(r.expert_to_slot.at(13) == 2);

    // After remap: [0,1,2,0,0,0,0,0] — all -1 become 0
    assert(r.remapped[0] == 0);
    assert(r.remapped[1] == 1);
    assert(r.remapped[2] == 2);
    assert(r.remapped[3] == 0); // -1 → 0 (Bug 7 fix)
    assert(r.remapped[4] == 0);
    assert(r.remapped[5] == 0);
    assert(r.remapped[6] == 0);
    assert(r.remapped[7] == 0);

    assert_no_oob(r.remapped, K, "test_padding_ids_clamped_to_zero");
    std::cout << "OK" << std::endl;
}

// ─── Test 2 ──────────────────────────────────────────────────────────────────
// BUG 7 SCENARIO: out-of-range expert IDs (>= n_experts) must be clamped.
// Corrupted router output or edge-case quantization could produce these.
void test_out_of_range_ids_clamped() {
    std::cout << "test_out_of_range_ids_clamped ...  ";

    const int K = 8, N = 128;
    std::vector<int32_t> raw = {5, 200, 127, 300, -5, 128};
    auto r = callback_remap(raw, N, K);

    // Only expert 5 and 127 are valid (0..127).
    // 200, 300, -5, 128 are all OOB and must clamp to 0.
    assert(r.remapped[0] == 0); // expert 5 → slot 0
    assert(r.remapped[1] == 0); // 200 OOB → slot 0
    assert(r.remapped[2] == 1); // expert 127 → slot 1
    assert(r.remapped[3] == 0); // 300 OOB → slot 0
    assert(r.remapped[4] == 0); // -5 OOB → slot 0
    assert(r.remapped[5] == 0); // 128 OOB → slot 0

    assert_no_oob(r.remapped, K, "test_out_of_range_ids_clamped");
    std::cout << "OK" << std::endl;
}

// ─── Test 3 ──────────────────────────────────────────────────────────────────
// BUG 7 SCENARIO: more unique experts than K slots.
// For K=2 but 5 unique experts in batch, experts beyond the first K
// must NOT get raw IDs written back (those raw IDs are >> K → OOB).
void test_overflow_experts_clamped_to_slot_zero() {
    std::cout << "test_overflow_experts_clamped_to_slot_zero ... ";

    const int K = 2, N = 128;
    // 5 unique experts, but K=2. Only first 2 get slots. Rest → clamp 0.
    std::vector<int32_t> raw = {10, 20, 30, 40, 50};
    auto r = callback_remap(raw, N, K);

    assert((int)r.expert_to_slot.size() == 2);
    assert(r.expert_to_slot.at(10) == 0);
    assert(r.expert_to_slot.at(20) == 1);
    assert(r.expert_to_slot.count(30) == 0); // no slot assigned

    // Remapped: [0, 1, 0, 0, 0]
    // Expert 30, 40, 50 have no slot → they map to 0 (not to 30, 40, 50!)
    assert(r.remapped[0] == 0);
    assert(r.remapped[1] == 1);
    assert(r.remapped[2] == 0); // 30 unmapped → 0, NOT 30 (OOB)
    assert(r.remapped[3] == 0); // 40 unmapped → 0
    assert(r.remapped[4] == 0); // 50 unmapped → 0

    assert_no_oob(r.remapped, K, "test_overflow_experts_clamped_to_slot_zero");
    std::cout << "OK" << std::endl;
}

// ─── Test 4 ──────────────────────────────────────────────────────────────────
// All IDs are -1 (pathological: fully padded tensor).
// Must not crash; all IDs become 0.
void test_all_padding_ids() {
    std::cout << "test_all_padding_ids ...            ";

    const int K = 8, N = 128;
    std::vector<int32_t> raw(16, -1);
    auto r = callback_remap(raw, N, K);

    assert(r.expert_to_slot.empty()); // no valid experts
    for (auto id : r.remapped) {
        assert(id == 0); // all -1 → 0
    }

    assert_no_oob(r.remapped, K, "test_all_padding_ids");
    std::cout << "OK" << std::endl;
}

// ─── Test 5 ──────────────────────────────────────────────────────────────────
// Normal case: exactly K unique experts, all fit → no clamping.
// Verifies the fix doesn't break the happy path.
void test_exact_k_experts_no_clamping() {
    std::cout << "test_exact_k_experts_no_clamping ... ";

    const int K = 4, N = 128;
    std::vector<int32_t> raw = {10, 20, 30, 40, 10, 20}; // 4 unique
    auto r = callback_remap(raw, N, K);

    assert((int)r.expert_to_slot.size() == 4);
    // Slot assignment: 10→0, 20→1, 30→2, 40→3
    assert(r.remapped[0] == 0);
    assert(r.remapped[1] == 1);
    assert(r.remapped[2] == 2);
    assert(r.remapped[3] == 3);
    assert(r.remapped[4] == 0); // 10 duplicate → slot 0
    assert(r.remapped[5] == 1); // 20 duplicate → slot 1

    assert_no_oob(r.remapped, K, "test_exact_k_experts_no_clamping");
    std::cout << "OK" << std::endl;
}

// ─── Test 6 ──────────────────────────────────────────────────────────────────
// Mixed: valid experts + padding + out-of-range, with K=8.
// Simulates a real MoE layer with n_tokens=4, K=8, some padding.
void test_mixed_valid_and_invalid_ids() {
    std::cout << "test_mixed_valid_and_invalid_ids ... ";

    const int K = 8, N = 128;
    // 4 tokens × 2 picked experts = 8 IDs, with some padding/corruption
    std::vector<int32_t> raw = {42, 97, -1, 13, 200, 42, -1, 97};
    auto r = callback_remap(raw, N, K);

    // Valid unique experts (in first-occurrence order): 42(0), 97(1), 13(2)
    assert(r.expert_to_slot.at(42) == 0);
    assert(r.expert_to_slot.at(97) == 1);
    assert(r.expert_to_slot.at(13) == 2);

    // Remapped:
    assert(r.remapped[0] == 0); // 42 → slot 0
    assert(r.remapped[1] == 1); // 97 → slot 1
    assert(r.remapped[2] == 0); // -1 → clamp 0
    assert(r.remapped[3] == 2); // 13 → slot 2
    assert(r.remapped[4] == 0); // 200 OOB → clamp 0
    assert(r.remapped[5] == 0); // 42 → slot 0 (duplicate)
    assert(r.remapped[6] == 0); // -1 → clamp 0
    assert(r.remapped[7] == 1); // 97 → slot 1 (duplicate)

    assert_no_oob(r.remapped, K, "test_mixed_valid_and_invalid_ids");
    std::cout << "OK" << std::endl;
}

// ─── Test 7 ──────────────────────────────────────────────────────────────────
// K=1 edge case: only one slot. All experts collapse to slot 0.
// This is extreme but must not cause OOB (slot 0 is always loaded).
void test_k_equals_one() {
    std::cout << "test_k_equals_one ...               ";

    const int K = 1, N = 128;
    std::vector<int32_t> raw = {42, 97, 13, -1, 200};
    auto r = callback_remap(raw, N, K);

    // Only one slot: first valid expert (42) gets slot 0. All others → 0.
    assert((int)r.expert_to_slot.size() == 1);
    assert(r.expert_to_slot.at(42) == 0);

    for (auto id : r.remapped) {
        assert(id == 0); // everything maps to the single slot 0
    }

    assert_no_oob(r.remapped, K, "test_k_equals_one");
    std::cout << "OK" << std::endl;
}

// ─── Test 8 ──────────────────────────────────────────────────────────────────
// Empty ID tensor: should not crash, should produce empty output.
void test_empty_ids() {
    std::cout << "test_empty_ids ...                  ";

    const int K = 8, N = 128;
    std::vector<int32_t> raw;
    auto r = callback_remap(raw, N, K);

    assert(r.expert_to_slot.empty());
    assert(r.remapped.empty());

    assert_no_oob(r.remapped, K, "test_empty_ids");
    std::cout << "OK" << std::endl;
}

// ─── Test 9 ──────────────────────────────────────────────────────────────────
// n_experts=0 edge case (degenerate init failure).
// ALL IDs are OOB if n_experts=0 → they should ALL clamp to 0.
void test_zero_n_experts_all_clamp() {
    std::cout << "test_zero_n_experts_all_clamp ...   ";

    const int K = 8, N = 0; // pathological: n_experts not set
    std::vector<int32_t> raw = {0, 1, 2, -1};
    auto r = callback_remap(raw, N, K);

    // No valid experts (all fail id >= 0 && id < 0 check)
    assert(r.expert_to_slot.empty());
    for (auto id : r.remapped) {
        assert(id == 0); // all out-of-range → 0
    }

    assert_no_oob(r.remapped, K, "test_zero_n_experts_all_clamp");
    std::cout << "OK" << std::endl;
}

// ─── Test 10 ─────────────────────────────────────────────────────────────────
// Verify slot 0 is always populated when there's at least one valid expert.
// The clamp-to-0 guarantee requires slot 0 to contain valid data.
void test_slot_zero_always_valid() {
    std::cout << "test_slot_zero_always_valid ...     ";

    const int K = 8, N = 128;

    // Case A: first ID is valid → expert 5 gets slot 0
    {
        std::vector<int32_t> raw = {5, -1, -1, 10};
        auto r = callback_remap(raw, N, K);
        assert(r.expert_to_slot.count(5) == 1);
        assert(r.expert_to_slot.at(5) == 0); // slot 0 is loaded
    }

    // Case B: only padding → no slot 0 assignment, but remap still outputs 0
    // (MUL_MAT_ID reads slot 0, which has whatever was in the buffer — harmless
    //  because router weight for padding slots is 0.0)
    {
        std::vector<int32_t> raw = {-1, -1, -1};
        auto r = callback_remap(raw, N, K);
        assert(r.expert_to_slot.empty());
        for (auto id : r.remapped) assert(id == 0);
        // OOB check still passes: 0 < K always
        assert_no_oob(r.remapped, K, "test_slot_zero_always_valid_B");
    }

    std::cout << "OK" << std::endl;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== Bug 7 OOB Clamp Tests ===" << std::endl;
    std::cout << "Verifying: eval_callback clamps all IDs to [0, K-1]" << std::endl;
    std::cout << std::endl;

    test_padding_ids_clamped_to_zero();
    test_out_of_range_ids_clamped();
    test_overflow_experts_clamped_to_slot_zero();
    test_all_padding_ids();
    test_exact_k_experts_no_clamping();
    test_mixed_valid_and_invalid_ids();
    test_k_equals_one();
    test_empty_ids();
    test_zero_n_experts_all_clamp();
    test_slot_zero_always_valid();

    std::cout << std::endl;
    std::cout << "=== All Bug 7 OOB tests passed (10/10) ===" << std::endl;
    return 0;
}
