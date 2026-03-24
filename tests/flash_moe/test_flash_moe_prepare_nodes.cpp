// test_flash_moe_prepare_nodes.cpp
//
// Validates the logic inside prepare_nodes() that was reworked in Step 2.2:
//   1. ID range filtering: IDs must satisfy 0 <= id < n_experts.
//      Negative IDs (-1 router padding) and out-of-range IDs (uninitialized
//      memory after ggml_backend_tensor_get) must both be dropped.
//   2. ID deduplication: the same expert can appear many times in one token
//      batch; ensure_expert_loaded() must be called at most once per unique id.
//   3. Filename construction: std::string concatenation never overflows,
//      regardless of how long the experts_dir path is.
//
// Build (standalone — no ggml link required):
//   clang++ -std=c++17 -I src -I ggml/include \
//     tests/flash_moe/test_flash_moe_prepare_nodes.cpp \
//     -o /tmp/test_prepare_nodes && /tmp/test_prepare_nodes

#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <string>
#include <cstdint>

// ─── Helpers that mirror the logic in prepare_nodes() ─────────────────────

// Returns the set of unique valid expert IDs from a raw id vector.
// Mirrors the filtering block in flash_moe_manager.cpp::prepare_nodes().
static std::unordered_set<int32_t> filter_ids(const std::vector<int32_t>& raw,
                                               int n_experts) {
    std::unordered_set<int32_t> out;
    for (auto id : raw) {
        if (id >= 0 && id < n_experts) out.insert(id);
    }
    return out;
}

// Mirrors the filename construction in flash_moe_manager.cpp::ensure_expert_loaded().
static std::string build_expert_path(const std::string& dir,
                                      int layer, int expert_id) {
    return dir + "/blk"
        + (layer     < 10 ? "0" : "") + std::to_string(layer)
        + "_exp"
        + (expert_id < 10 ? "00" : (expert_id < 100 ? "0" : ""))
        + std::to_string(expert_id) + ".bin";
}

// ─── Test 1 ───────────────────────────────────────────────────────────────
// Negative IDs (router padding) and IDs >= n_experts must both be dropped.
void test_id_range_filter() {
    std::cout << "Running test_id_range_filter..." << std::endl;

    const int n_experts = 8;
    // Typical token batch: 4 valid IDs, 2 padded (-1), 2 out-of-range (8,99)
    std::vector<int32_t> raw = {3, -1, 0, 8, 7, -1, 99, 3};
    auto ids = filter_ids(raw, n_experts);

    // Valid: 0, 3, 7
    assert(ids.size() == 3);
    assert(ids.count(0) == 1);
    assert(ids.count(3) == 1);
    assert(ids.count(7) == 1);
    // Must be excluded
    assert(ids.count(-1) == 0);
    assert(ids.count(8)  == 0);
    assert(ids.count(99) == 0);

    std::cout << "test_id_range_filter passed!" << std::endl;
}

// ─── Test 2 ───────────────────────────────────────────────────────────────
// All IDs invalid → empty set → no expert loads attempted.
void test_all_invalid_ids() {
    std::cout << "Running test_all_invalid_ids..." << std::endl;

    const int n_experts = 4;
    std::vector<int32_t> raw = {-1, -1, 4, 100, -2};
    auto ids = filter_ids(raw, n_experts);
    assert(ids.empty());

    std::cout << "test_all_invalid_ids passed!" << std::endl;
}

// ─── Test 3 ───────────────────────────────────────────────────────────────
// All IDs are the same expert → deduplication yields exactly one entry.
void test_deduplication() {
    std::cout << "Running test_deduplication..." << std::endl;

    const int n_experts = 64;
    std::vector<int32_t> raw(32, 5); // 32 copies of expert 5
    auto ids = filter_ids(raw, n_experts);
    assert(ids.size() == 1);
    assert(ids.count(5) == 1);

    std::cout << "test_deduplication passed!" << std::endl;
}

// ─── Test 4 ───────────────────────────────────────────────────────────────
// Boundary values: id == 0 and id == n_experts-1 are valid;
// id == n_experts is not.
void test_boundary_ids() {
    std::cout << "Running test_boundary_ids..." << std::endl;

    const int n_experts = 128;
    std::vector<int32_t> raw = {0, n_experts - 1, n_experts};
    auto ids = filter_ids(raw, n_experts);

    assert(ids.size() == 2);
    assert(ids.count(0)             == 1);
    assert(ids.count(n_experts - 1) == 1);
    assert(ids.count(n_experts)     == 0);

    std::cout << "test_boundary_ids passed!" << std::endl;
}

// ─── Test 5 ───────────────────────────────────────────────────────────────
// Filename construction: zero-padding must match the pattern blkNN_expNNN.bin,
// and a long directory path must not cause truncation.
void test_filename_construction() {
    std::cout << "Running test_filename_construction..." << std::endl;

    // Single-digit layer, single-digit expert → zero-padded to 2+3 digits
    assert(build_expert_path("/tmp", 0, 0)   == "/tmp/blk00_exp000.bin");
    assert(build_expert_path("/tmp", 9, 9)   == "/tmp/blk09_exp009.bin");

    // Double-digit layer, double-digit expert
    assert(build_expert_path("/tmp", 10,  10) == "/tmp/blk10_exp010.bin");
    assert(build_expert_path("/tmp", 99,  99) == "/tmp/blk99_exp099.bin");

    // Triple-digit expert
    assert(build_expert_path("/tmp", 5, 100) == "/tmp/blk05_exp100.bin");
    assert(build_expert_path("/tmp", 5, 127) == "/tmp/blk05_exp127.bin");

    // Very long directory path — std::string never overflows
    std::string long_dir(240, 'x');
    std::string path = build_expert_path(long_dir, 63, 255);
    assert(path.size() > 240);
    assert(path.find("blk63_exp255.bin") != std::string::npos);

    std::cout << "test_filename_construction passed!" << std::endl;
}

// ─── Test 6 ───────────────────────────────────────────────────────────────
// n_experts == 0 → any ID is out of range → empty set.
void test_zero_experts() {
    std::cout << "Running test_zero_experts..." << std::endl;

    std::vector<int32_t> raw = {0, 1, 2};
    auto ids = filter_ids(raw, 0);
    assert(ids.empty());

    std::cout << "test_zero_experts passed!" << std::endl;
}

int main() {
    test_id_range_filter();
    test_all_invalid_ids();
    test_deduplication();
    test_boundary_ids();
    test_filename_construction();
    test_zero_experts();
    std::cout << "\nAll prepare_nodes logic tests passed!" << std::endl;
    return 0;
}
