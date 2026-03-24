// test_flash_moe_slot_remap.cpp
//
// Validates the Phase 2.6 slot-based ID remapping logic in prepare_nodes():
//
//   1. Unique expert IDs are assigned contiguous slot indices 0..K-1 in
//      first-occurrence order across the batch.
//   2. The IDs vector is remapped from raw expert IDs to slot indices.
//   3. Out-of-range and negative IDs remain unchanged (router padding = -1).
//   4. When the number of unique IDs exceeds K (= n_expert_used) the extra
//      IDs are capped and those positions get index -1 in the remapped output.
//   5. Duplicate IDs in the same batch resolve to the same slot.
//
// Build (standalone — no ggml link required):
//   g++ -std=c++17 tests/flash_moe/test_flash_moe_slot_remap.cpp \
//       -o /tmp/test_slot_remap && /tmp/test_slot_remap

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

// ─── Mirrors the slot-assignment + ID-remapping logic in prepare_nodes() ─────

struct SlotResult {
    std::unordered_map<int32_t, int32_t> expert_to_slot; // expert_id → slot
    std::vector<int32_t> remapped;                        // IDs after remap
};

static SlotResult build_slots_and_remap(const std::vector<int32_t>& raw_ids,
                                         int n_experts, int max_slots) {
    SlotResult r;
    int32_t slot = 0;

    // Assign slots in first-occurrence order.
    for (auto id : raw_ids) {
        if (id >= 0 && id < n_experts &&
            r.expert_to_slot.find(id) == r.expert_to_slot.end()) {
            if (slot >= max_slots) break;  // cap at K
            r.expert_to_slot[id] = slot++;
        }
    }

    // Remap IDs.
    r.remapped = raw_ids;
    for (auto& id : r.remapped) {
        if (id >= 0 && id < n_experts) {
            auto it = r.expert_to_slot.find(id);
            id = (it != r.expert_to_slot.end()) ? it->second : -1;
        }
    }
    return r;
}

// ─── Test 1 ──────────────────────────────────────────────────────────────────
// Basic case: 4 unique experts in a batch of 8, K=8 (no cap).
void test_basic_remap() {
    std::cout << "Running test_basic_remap..." << std::endl;

    std::vector<int32_t> raw = {42, 97, 13, 4, 42, 97, 13, 4};
    auto r = build_slots_and_remap(raw, 128, 8);

    // Slot assignment order: 42→0, 97→1, 13→2, 4→3
    assert(r.expert_to_slot.at(42) == 0);
    assert(r.expert_to_slot.at(97) == 1);
    assert(r.expert_to_slot.at(13) == 2);
    assert(r.expert_to_slot.at(4)  == 3);

    // Remapped IDs
    std::vector<int32_t> expected = {0, 1, 2, 3, 0, 1, 2, 3};
    assert(r.remapped == expected);

    std::cout << "test_basic_remap passed!" << std::endl;
}

// ─── Test 2 ──────────────────────────────────────────────────────────────────
// Negative / out-of-range IDs stay unchanged after remap.
void test_padding_ids_unchanged() {
    std::cout << "Running test_padding_ids_unchanged..." << std::endl;

    // -1 = router padding, 200 = out-of-range for n_experts=128
    std::vector<int32_t> raw = {-1, 5, 200, 5, -1};
    auto r = build_slots_and_remap(raw, 128, 8);

    assert(r.expert_to_slot.size() == 1);      // only expert 5 is valid
    assert(r.expert_to_slot.at(5) == 0);

    assert(r.remapped[0] == -1);   // unchanged
    assert(r.remapped[1] == 0);    // expert 5 → slot 0
    assert(r.remapped[2] == 200);  // out-of-range, unchanged
    assert(r.remapped[3] == 0);    // expert 5 → slot 0 again
    assert(r.remapped[4] == -1);   // padding

    std::cout << "test_padding_ids_unchanged passed!" << std::endl;
}

// ─── Test 3 ──────────────────────────────────────────────────────────────────
// Cap at K: batch has 10 unique experts but K=8.
// Extra experts (beyond K) are assigned no slot → remapped to -1.
void test_cap_at_k() {
    std::cout << "Running test_cap_at_k..." << std::endl;

    // 10 unique experts: 0..9.  K=8 → only 0..7 get slots; 8 and 9 do not.
    std::vector<int32_t> raw = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto r = build_slots_and_remap(raw, 128, 8);

    assert((int)r.expert_to_slot.size() == 8);

    for (int i = 0; i < 8; ++i) {
        assert(r.expert_to_slot.count(i) == 1);
        assert(r.remapped[i] == i);  // slot i for expert i
    }
    // Experts 8 and 9 were not assigned slots → remapped to -1.
    assert(r.remapped[8] == -1);
    assert(r.remapped[9] == -1);

    std::cout << "test_cap_at_k passed!" << std::endl;
}

// ─── Test 4 ──────────────────────────────────────────────────────────────────
// Single token: exactly K=8 unique experts, each used once.
void test_single_token_exact_k() {
    std::cout << "Running test_single_token_exact_k..." << std::endl;

    std::vector<int32_t> raw = {10, 20, 30, 40, 50, 60, 70, 80};
    auto r = build_slots_and_remap(raw, 128, 8);

    assert((int)r.expert_to_slot.size() == 8);
    for (int i = 0; i < 8; ++i) {
        int eid = raw[i];
        assert(r.expert_to_slot.at(eid) == i);
        assert(r.remapped[i] == i);
    }

    std::cout << "test_single_token_exact_k passed!" << std::endl;
}

// ─── Test 5 ──────────────────────────────────────────────────────────────────
// All IDs the same expert → one slot used, rest of batch maps to slot 0.
void test_all_same_expert() {
    std::cout << "Running test_all_same_expert..." << std::endl;

    std::vector<int32_t> raw(16, 77);  // 16 tokens, all using expert 77
    auto r = build_slots_and_remap(raw, 128, 8);

    assert(r.expert_to_slot.size() == 1);
    assert(r.expert_to_slot.at(77) == 0);

    for (auto id : r.remapped) {
        assert(id == 0);
    }

    std::cout << "test_all_same_expert passed!" << std::endl;
}

// ─── Test 6 ──────────────────────────────────────────────────────────────────
// Boundary: expert_id == n_experts-1 is valid; == n_experts is not.
void test_boundary_expert_ids() {
    std::cout << "Running test_boundary_expert_ids..." << std::endl;

    const int N = 128;
    std::vector<int32_t> raw = {0, N - 1, N};
    auto r = build_slots_and_remap(raw, N, 8);

    assert(r.expert_to_slot.count(0)     == 1);
    assert(r.expert_to_slot.count(N - 1) == 1);
    assert(r.expert_to_slot.count(N)     == 0);  // out of range

    assert(r.remapped[0] == 0);         // expert 0   → slot 0
    assert(r.remapped[1] == 1);         // expert 127 → slot 1
    assert(r.remapped[2] == (int32_t)N); // N=128 unchanged (not in map, but also > 0)

    std::cout << "test_boundary_expert_ids passed!" << std::endl;
}

// ─── Test 7 ──────────────────────────────────────────────────────────────────
// Slot indices are stable across two projections (gate and down) for the same
// layer.  Both must use the same expert_to_slot map so indices are consistent.
void test_slot_consistency_across_projections() {
    std::cout << "Running test_slot_consistency_across_projections..." << std::endl;

    // Simulate: same IDs seen for gate and down projections (same layer).
    std::vector<int32_t> raw = {5, 13, 99, 5, 13};

    // Both projections should see the SAME mapping.
    auto r_gate = build_slots_and_remap(raw, 128, 8);
    auto r_down = build_slots_and_remap(raw, 128, 8);

    // Deterministic first-occurrence assignment: 5→0, 13→1, 99→2
    assert(r_gate.expert_to_slot == r_down.expert_to_slot);
    assert(r_gate.remapped       == r_down.remapped);

    std::cout << "test_slot_consistency_across_projections passed!" << std::endl;
}

int main() {
    test_basic_remap();
    test_padding_ids_unchanged();
    test_cap_at_k();
    test_single_token_exact_k();
    test_all_same_expert();
    test_boundary_expert_ids();
    test_slot_consistency_across_projections();

    std::cout << "\nAll slot remap tests passed!" << std::endl;
    return 0;
}
