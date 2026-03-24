// tests/flash_moe/test_flash_moe_integration_wiring.cpp
#include "flash_moe/flash_moe_manager.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void create_dummy_index(const std::string& dir) {
    fs::create_directories(dir);
    std::ofstream f(dir + "/expert_index.json");
    f << R"({
        "n_layers": 2,
        "n_experts": 8,
        "experts": {
            "0_0": { "file_size": 32768, "gate_offset": 0, "gate_bytes": 1024, "up_offset": 1024, "up_bytes": 1024, "down_offset": 2048, "down_bytes": 1024 },
            "1_0": { "file_size": 32768, "gate_offset": 0, "gate_bytes": 1024, "up_offset": 1024, "up_bytes": 1024, "down_offset": 2048, "down_bytes": 1024 }
        }
    })";
}

int main() {
    std::string test_dir = "/tmp/fmoe_test_wiring";
    create_dummy_index(test_dir);

    auto& mgr = FlashMoE::get_manager();
    
    std::cout << "Testing FlashMoE initialization..." << std::endl;
    mgr.init(test_dir, 1024);
    assert(mgr.enabled == true);
    assert(mgr.experts_dir == test_dir);
    assert(mgr.cache_size_mib == 1024);
    std::cout << "Init with valid path passed!" << std::endl;

    std::cout << "Testing FlashMoE initialization with invalid path..." << std::endl;
    mgr.init("/non/existent/path", 1024);
    assert(mgr.enabled == false);
    std::cout << "Init with invalid path passed!" << std::endl;

    fs::remove_all(test_dir);
    std::cout << "All wiring tests passed!" << std::endl;
    return 0;
}
