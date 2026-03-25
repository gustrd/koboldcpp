// test_flash_moe_eval_callback.cpp
//
// Linked tests for ExpertManager::eval_callback and prepare_nodes.
// These link against flash_moe_manager.cpp + ggml_stubs.cpp.
//
// Tests updated for the ordering fix: callback now triggers on
// "ffn_moe_weights-N" (after GET_ROWS) instead of "ffn_moe_topk-N" (before).
// IDs are read/written from ids_tensor (= ffn_moe_topk stored in
// current_split_layers[layer].ids_tensor), NOT from t (= ffn_moe_weights).
//
// Build:
//   clang++ -std=c++17 -I src -I ggml/include -I vendor \
//     tests/flash_moe/test_flash_moe_eval_callback.cpp \
//     tests/flash_moe/ggml_stubs.cpp \
//     src/flash_moe/flash_moe_manager.cpp \
//     src/flash_moe/flash_moe_cache.cpp \
//     src/flash_moe/flash_moe_platform.cpp \
//     -o /tmp/test_eval_callback && /tmp/test_eval_callback

#include "../../src/flash_moe/flash_moe_manager.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace FlashMoE;

extern "C" {
    extern void (*g_mock_tensor_set)(ggml_tensor*, const void*, size_t, size_t);
    extern void (*g_mock_tensor_get)(const ggml_tensor*, void*, size_t, size_t);
}

// Per-tensor data storage: tensor pointer → int32 data buffer.
// Allows get/set mocks to route to the correct tensor's data.
std::unordered_map<const ggml_tensor*, std::vector<int32_t>> g_tensor_data;

static void my_tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    auto it = g_tensor_data.find(tensor);
    if (it != g_tensor_data.end())
        memcpy(data, (const char*)it->second.data() + offset, size);
}

static void my_tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    auto it = g_tensor_data.find(tensor);
    if (it != g_tensor_data.end())
        memcpy((char*)it->second.data() + offset, data, size);
}

// Write a minimal expert_index.json to dir so mgr.init() succeeds and
// g_layers[*].n_experts is populated.
static void write_minimal_index(const char* dir, int n_layers, int n_experts, int k) {
    // Build JSON manually
    std::ofstream f(std::string(dir) + "/expert_index.json");
    f << "{\n";
    f << "  \"config\": {"
      << "\"n_layers\":" << n_layers << ","
      << "\"n_experts\":" << n_experts << ","
      << "\"n_expert_used\":" << k
      << "},\n";
    f << "  \"experts\": {\n";
    // One entry per layer (expert 0) — sizes don't matter for remap tests
    for (int l = 0; l < n_layers; ++l) {
        f << "    \"" << l << "_0\": {"
          << "\"file\":\"blk00_exp000.bin\","
          << "\"file_size\":4096,"
          << "\"gate_offset\":0,\"gate_bytes\":1024,"
          << "\"up_offset\":1024,\"up_bytes\":1024,"
          << "\"down_offset\":2048,\"down_bytes\":1024,"
          << "\"dtype\":\"Q4_K\","
          << "\"gate_shape\":[128,8],\"up_shape\":[128,8],\"down_shape\":[8,128]"
          << "}";
        if (l + 1 < n_layers) f << ",";
        f << "\n";
    }
    f << "  }\n}\n";
}

// ─── Test 1: Ask phase trigger ────────────────────────────────────────────────
// Fixed: trigger on "ffn_moe_weights-N", NOT "ffn_moe_topk-N".
void test_callback_ask_phase() {
    std::cout << "Testing eval_callback ask phase (ordering fix)..." << std::endl;

    ggml_tensor t;

    // Must trigger: GET_ROWS output, fires AFTER GET_ROWS uses original IDs
    strcpy(t.name, "ffn_moe_weights-0");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == true);

    strcpy(t.name, "ffn_moe_weights-23");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == true);

    // Must NOT trigger: ARGSORT output — firing here would corrupt GET_ROWS
    strcpy(t.name, "ffn_moe_topk-0");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == false);

    strcpy(t.name, "ffn_moe_topk-23");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == false);

    // Must NOT trigger: weight norm runs after GET_ROWS, no need to intercept again
    strcpy(t.name, "ffn_moe_weights_norm-0");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == false);

    // Must NOT trigger: unrelated tensors
    strcpy(t.name, "other_tensor");
    assert(ExpertManager::eval_callback(&t, true, nullptr) == false);

    std::cout << "  OK" << std::endl;
}

// ─── Test 2: Remap reads from ids_tensor, not from t ─────────────────────────
// t is ffn_moe_weights (F32 probs), ids_tensor is ffn_moe_topk (I32 expert IDs).
void test_callback_remap_reads_ids_tensor() {
    std::cout << "Testing eval_callback remap reads from ids_tensor..." << std::endl;

    // Write JSON so g_layers[0].n_experts = 128
    write_minimal_index("/tmp/mock_eval_cb", 1, 128, 8);

    ExpertManager& mgr = get_manager();
    mgr.init("/tmp/mock_eval_cb", 128);
    mgr.enabled = true;
    mgr.n_expert_used = 8;
    mgr.current_split_layers.clear();

    // t = ffn_moe_weights (the GET_ROWS output) — callback trigger
    ggml_tensor t_weights;
    strcpy(t_weights.name, "ffn_moe_weights-0");
    t_weights.ne[0] = 3; t_weights.ne[1] = 1; t_weights.ne[2] = 1; t_weights.ne[3] = 1;

    // ids_tensor = ffn_moe_topk — holds expert IDs to remap
    ggml_tensor t_ids;
    strcpy(t_ids.name, "ffn_moe_topk-0");
    t_ids.ne[0] = 3; t_ids.ne[1] = 1; t_ids.ne[2] = 1; t_ids.ne[3] = 1;

    // IDs: [3, 7, 3] → unique: {3→0, 7→1} → remapped: [0, 1, 0]
    g_tensor_data[&t_ids] = {3, 7, 3};

    // Set up current_split_layers: ids_tensor must point to t_ids
    ggml_tensor w;
    strcpy(w.name, "blk.0.ffn_gate_exps.weight");
    w.ne[0] = 128; w.ne[1] = 8; w.ne[2] = 128; w.ne[3] = 1;
    mgr.current_split_layers[0].weight_tensors.push_back(&w);
    mgr.current_split_layers[0].ids_tensor = &t_ids;

    bool cont = ExpertManager::eval_callback(&t_weights, false, nullptr);
    assert(cont == true);

    auto& result = g_tensor_data[&t_ids];
    assert(result[0] == 0);  // 3 → slot 0
    assert(result[1] == 1);  // 7 → slot 1
    assert(result[2] == 0);  // 3 → slot 0 (duplicate)

    std::cout << "  OK" << std::endl;
}

// ─── Test 3: Invalid IDs clamped to slot 0 (Bug 7 regression) ────────────────
void test_callback_remap_clamping() {
    std::cout << "Testing eval_callback remap clamping (Bug 7)..." << std::endl;

    ExpertManager& mgr = get_manager();
    mgr.enabled = true;
    mgr.n_expert_used = 8;
    mgr.current_split_layers.clear();

    ggml_tensor t_weights;
    strcpy(t_weights.name, "ffn_moe_weights-0");
    t_weights.ne[0] = 3; t_weights.ne[1] = 1; t_weights.ne[2] = 1; t_weights.ne[3] = 1;

    ggml_tensor t_ids;
    strcpy(t_ids.name, "ffn_moe_topk-0");
    t_ids.ne[0] = 3; t_ids.ne[1] = 1; t_ids.ne[2] = 1; t_ids.ne[3] = 1;

    // -1 is padding (router), must clamp to 0
    g_tensor_data[&t_ids] = {3, -1, 7};

    ggml_tensor w;
    strcpy(w.name, "blk.0.ffn_gate_exps.weight");
    w.ne[0] = 128; w.ne[1] = 8; w.ne[2] = 128; w.ne[3] = 1;
    mgr.current_split_layers[0].weight_tensors.push_back(&w);
    mgr.current_split_layers[0].ids_tensor = &t_ids;

    ExpertManager::eval_callback(&t_weights, false, nullptr);

    auto& result = g_tensor_data[&t_ids];
    assert(result[0] == 0);  // 3 → slot 0
    assert(result[1] == 0);  // -1 → clamped to 0
    assert(result[2] == 1);  // 7 → slot 1

    std::cout << "  OK" << std::endl;
}

// ─── Test 4: prepare_nodes indexes weight + ids tensors correctly ─────────────
void test_prepare_nodes_index() {
    std::cout << "Testing prepare_nodes index pass..." << std::endl;

    ExpertManager& mgr = get_manager();
    mgr.enabled = true;
    mgr.current_split_layers.clear();

    ggml_tensor w1 = {}, w2 = {}, ids = {};
    strcpy(w1.name, "blk.5.ffn_gate_exps.weight");
    strcpy(w2.name, "blk.5.ffn_up_exps.weight");
    strcpy(ids.name, "ffn_moe_topk-5");
    // Initialize ne[] so ggml_nelements returns a sensible value
    ids.ne[0] = 3; ids.ne[1] = 1; ids.ne[2] = 1; ids.ne[3] = 1;

    // Provide mock data for ids tensor (real expert IDs)
    g_tensor_data[&ids] = {42, 99, 42};

    ggml_tensor node = {}, node2 = {};
    node.op  = GGML_OP_MUL_MAT_ID; node.src[0]  = &w1; node.src[2]  = &ids;
    node2.op = GGML_OP_MUL_MAT_ID; node2.src[0] = &w2; node2.src[2] = &ids;

    ggml_tensor* nodes[] = { &node, &node2 };
    mgr.prepare_nodes(nodes, 2);

    assert(mgr.current_split_layers.size() == 1);
    assert(mgr.current_split_layers.count(5) == 1);
    assert(mgr.current_split_layers[5].weight_tensors.size() == 2);
    assert(mgr.current_split_layers[5].ids_tensor == &ids);

    // Clean up mock data
    g_tensor_data.erase(&ids);

    std::cout << "  OK" << std::endl;
}

int main() {
    // Use mkdir -p equivalent via system() for the temp dir
    system("mkdir -p /tmp/mock_eval_cb");

    g_mock_tensor_get = my_tensor_get;
    g_mock_tensor_set = my_tensor_set;

    try {
        test_callback_ask_phase();
        test_callback_remap_reads_ids_tensor();
        test_callback_remap_clamping();
        test_prepare_nodes_index();
        std::cout << "\nAll eval_callback tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
