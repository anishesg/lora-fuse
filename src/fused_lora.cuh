#pragma once
#include <cuda_fp16.h>
#include "lora_config.cuh"

// Single-kernel fused base+LoRA matvec.
// Single pass tiles d_in: loads W columns and A columns simultaneously,
// accumulates W*x in fp32 registers and A*x in r fp32 registers.
// B*(A*x) epilogue adds scaled result before writing fp16 output.
//
// TILE_K is tuned per rank:
//   rank 8,16  -> TILE_K=128 (small sA allows wider tile for W loading bandwidth)
//   rank 32,64 -> TILE_K=64  (sA grows, keep shared memory under 16 KB)
// Template param RANK must be known at compile time.
// Use launch_fused_lora() for runtime dispatch by cfg.rank.
template <int RANK, int TILE_K = 64, int TILE_ROWS = 128>
__global__ __launch_bounds__(TILE_ROWS) void fused_lora_kernel(
    const __half* __restrict__ W,    // [d_out, d_in]
    const __half* __restrict__ A,    // [RANK, d_in]
    const __half* __restrict__ B,    // [d_out, RANK]
    const __half* __restrict__ x,    // [d_in]
    __half* __restrict__ y,          // [d_out]
    int d_out,
    int d_in,
    float alpha);

// Runtime dispatch: selects template specialization based on cfg.rank.
void launch_fused_lora(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* x,
    __half* y,
    int adapter_id,
    cudaStream_t stream = nullptr);
