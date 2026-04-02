# Flash-MoE: Feature Plan & Strategy

## 1. Objective & Big Idea
Run large Mixture-of-Experts models (e.g., Qwen3-30B, Qwen3.5-397B, gpt-oss-120b) via `llama.cpp` using a native zero-swap inference engine. By bypassing OS-level page cache thrashing, expert weights remain on SSD until needed and are efficiently loaded into a minimal pre-allocated memory pool.

## 2. Architectural Strategy
- **Two-Tier Caching:** Features an LRU cache and a heat map tracker with exponential decay to continuously pre-pin the "hottest" experts, practically eliminating redundant SSD reads.
- **K-Slot Mapping:** Expert tensors are radically shrunk to accommodate only the `K` active experts per layer (e.g., `ne[2]=K` instead of 128). An eval callback transparently remaps expert IDs in-place, plunging memory overhead from ~12GB down to <1GB.
- **Eval Callback Pipeline:** A native ggml callback mechanism fires sequentially between the router and MoE ops (GET_ROWS → SSD Read → MUL_MAT_ID). This serial execution design is hardware-optimal for unified memory SoCs.
- **Single-Token Batching Constraint:** `n_batch = 1` is enforced during prompt evaluation. This bounds the maximum unique experts required per layer to exactly `K`, mathematically preventing slot buffer overflow.

## 3. Current State & Performance
**Current Status:** Core integration is stable across CPU inference (`--gpulayers 0`) and Vulkan/Metal backends. K-Slot mapping, file handle pooling, bias remapping, and cross-platform hooks are reliably implemented.

**Performance Baseline (Intel Lunar Lake iGPU / Windows Vulkan, Qwen3-30B):**
| Hardware | Model | Speed |
| :--- | :--- | :--- |
| **Mac M2 (16GB)** | Qwen3-30B-A3B | **3.5 T/s** |
| **Intel Lunar Lake (32GB)** | GPT-OSS-120B | **1.5 T/s** |
- **Cache Efficiency:** Achieves 70%—98% cache hit rates post-warmup due to aggressive heat map pre-pinning, depending of the memory size defined at startup.

## 4. Strategic Next Steps & Backlog

| Priority | Initiative | Strategic Goal | Status |
|----------|------------|----------------|--------|
| **P1** | **Full GPU Offloading for MoE** | Eliminate Vulkan-to-CPU synchronization bottleneck. Transition expert tensor allocation onto Vulkan GPU buffers so `MUL_MAT_ID` runs natively on the GPU alongside Attention ops. | Pending |
| **P2** | **Prompt Eval Acceleration** | Single-batch limits prompt processing to ~2.1 T/s. Investigate small-batch hybrid strategies, chat template pre-warming, or speculative expert routing to drastically speed up prompt ingestion. | Pending |
| **P3** | **Linux O_DIRECT I/O** | Implement unbuffered SSD reads on Linux to match Windows (`FILE_FLAG_NO_BUFFERING`) and macOS (`F_NOCACHE`), preventing page cache pollution and memory bloat. | Pending |
| **P4** | **LRU Cache Tuning & CLI Params** | Expose `--flashmoecache <MiB>` and `--flashmoedecay <alpha>` to empower end-users to balance their hit-rate vs system RAM usage tradeoffs. | Pending |
| **P5** | **GPU-Side Expert Pinning** | Eliminate the remaining CPU → GPU PCIe copies entirely for perpetually hot experts in the highest cache tier. | Pending |
