// Minimal ggml stubs for Flash-MoE unit tests that link flash_moe_manager.o
// but do not exercise ggml backend paths.
#include "ggml.h"
#include "ggml-backend.h"

extern "C" {

void (*g_mock_tensor_set)(ggml_tensor*, const void*, size_t, size_t) = nullptr;
void (*g_mock_tensor_get)(const ggml_tensor*, void*, size_t, size_t) = nullptr;

void ggml_backend_tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (g_mock_tensor_set) {
        g_mock_tensor_set(tensor, data, offset, size);
        return;
    }
}

void ggml_backend_tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (g_mock_tensor_get) {
        g_mock_tensor_get(tensor, data, offset, size);
        return;
    }
}

int64_t ggml_nelements(const ggml_tensor* tensor) {
    if (!tensor) return 0;
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

} // extern "C"
