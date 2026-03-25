#include "../../src/flash_moe/flash_moe_manager.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <unordered_map>

using namespace FlashMoE;


extern "C" {
    extern void (*g_mock_tensor_set)(ggml_tensor*, const void*, size_t, size_t);
    extern void (*g_mock_tensor_get)(const ggml_tensor*, void*, size_t, size_t);
}

// Global mocks for get/set
std::vector<int32_t> g_mock_ids;
void mock_set_ids(const std::vector<int32_t>& ids) { g_mock_ids = ids; }

void my_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    (void)tensor; (void)offset;
    memcpy(data, g_mock_ids.data(), size);
}

void my_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    (void)tensor; (void)offset;
    memcpy(g_mock_ids.data(), data, size);
}

void test_callback_ask_phase() {
    std::cout << "Testing eval_callback ask phase..." << std::endl;
    
    ggml_tensor t1;
    strcpy(t1.name, "ffn_moe_topk-0");
    assert(ExpertManager::eval_callback(&t1, true, nullptr) == true);
    
    ggml_tensor t2;
    strcpy(t2.name, "other_tensor");
    assert(ExpertManager::eval_callback(&t2, true, nullptr) == false);
    
    std::cout << "  OK" << std::endl;
}

void test_callback_remap_basic() {
    std::cout << "Testing eval_callback remap basic..." << std::endl;
    
    ExpertManager& mgr = get_manager();
    // Use a temporary experts dir for init
    mgr.init("tests/flash_moe/mock_experts", 128); 
    // Note: this might fail if dir doesn't exist, but we only need it to set enabled=true
    // and n_expert_used. Let's force them for the test if init fails.
    mgr.enabled = true;
    mgr.n_expert_used = 8;
    mgr.n_experts = 128;
    mgr.n_layers = 1;

    ggml_tensor t;
    strcpy(t.name, "ffn_moe_topk-0");
    t.ne[0] = 3; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    
    // IDs: [3, 7, 3] -> unique are {3, 7}. Map: 3->0, 7->1. Remapped: [0, 1, 0]
    mock_set_ids({3, 7, 3});
    
    // We need a weight tensor in current_split_layers to avoid early return
    ggml_tensor w;
    strcpy(w.name, "blk.0.ffn_gate_exps.weight");
    mgr.current_split_layers[0].weight_tensors.push_back(&w);
    
    // Note: eval_callback will try to call ensure_expert_loaded which calls g_cache.
    // If g_cache is null it prints error and continues. That's fine for testing remap logic.
    
    bool cont = ExpertManager::eval_callback(&t, false, nullptr);
    assert(cont == true);
    
    assert(g_mock_ids[0] == 0);
    assert(g_mock_ids[1] == 1);
    assert(g_mock_ids[2] == 0);
    
    std::cout << "  OK" << std::endl;
}

void test_callback_remap_padding() {
    std::cout << "Testing eval_callback remap padding (Bug 7 fix)..." << std::endl;
    
    ExpertManager& mgr = get_manager();
    mgr.enabled = true;
    mgr.n_expert_used = 8;
    
    ggml_tensor t;
    strcpy(t.name, "ffn_moe_topk-0");
    t.ne[0] = 3; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    
    // IDs: [3, -1, 7] -> -1 should be clamped to 0. Remapped: [0, 0, 1]
    mock_set_ids({3, -1, 7});
    
    ExpertManager::eval_callback(&t, false, nullptr);
    
    assert(g_mock_ids[0] == 0); // 3 -> slot 0
    assert(g_mock_ids[1] == 0); // -1 -> clamp 0
    assert(g_mock_ids[2] == 1); // 7 -> slot 1
    
    std::cout << "  OK" << std::endl;
}

void test_callback_remap_overflow() {
    std::cout << "Testing eval_callback remap overflow (>K experts)..." << std::endl;
    
    ExpertManager& mgr = get_manager();
    mgr.n_expert_used = 2; // Only 2 slots
    
    ggml_tensor t;
    strcpy(t.name, "ffn_moe_topk-0");
    t.ne[0] = 4; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    
    // IDs: [10, 20, 30, 10] -> slots: 10->0, 20->1, 30->clamp 0. Remapped: [0, 1, 0, 0]
    mock_set_ids({10, 20, 30, 10});
    
    ExpertManager::eval_callback(&t, false, nullptr);
    
    assert(g_mock_ids[0] == 0);
    assert(g_mock_ids[1] == 1);
    assert(g_mock_ids[2] == 0); // 30 overflowed slots 0,1 -> clamp 0
    assert(g_mock_ids[3] == 0); // 10 is slot 0
    
    std::cout << "  OK" << std::endl;
}

void test_prepare_nodes_index() {
    std::cout << "Testing prepare_nodes index pass..." << std::endl;
    
    ExpertManager& mgr = get_manager();
    mgr.enabled = true;
    mgr.current_split_layers.clear();
    
    ggml_tensor w1, w2, ids;
    strcpy(w1.name, "blk.5.ffn_gate_exps.weight");
    strcpy(w2.name, "blk.5.ffn_up_exps.weight");
    strcpy(ids.name, "ids_tensor");
    
    ggml_tensor node;
    node.op = GGML_OP_MUL_MAT_ID;
    node.src[0] = &w1;
    node.src[2] = &ids;
    
    ggml_tensor node2;
    node2.op = GGML_OP_MUL_MAT_ID;
    node2.src[0] = &w2;
    node2.src[2] = &ids;
    
    ggml_tensor* nodes[] = { &node, &node2 };
    mgr.prepare_nodes(nodes, 2);
    
    assert(mgr.current_split_layers.size() == 1);
    assert(mgr.current_split_layers.count(5) == 1);
    assert(mgr.current_split_layers[5].weight_tensors.size() == 2);
    assert(mgr.current_split_layers[5].ids_tensor == &ids);
    
    std::cout << "  OK" << std::endl;
}

int main() {
    try {
        g_mock_tensor_get = my_tensor_get;
        g_mock_tensor_set = my_tensor_set;

        test_callback_ask_phase();
        test_callback_remap_basic();
        test_callback_remap_padding();
        test_callback_remap_overflow();
        test_prepare_nodes_index();
        std::cout << "All eval_callback tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
