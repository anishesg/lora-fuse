#pragma once
#include <cuda_fp16.h>
#include "lora_config.cuh"

// Multi-adapter batched fused kernel.
// Input: X [batch, d_in], adapter_ids [batch] int32.
// Output: Y [batch, d_out].
//
// One thread block handles one token. Threads within the block cooperate on
// the output rows for that token. W is shared across all tokens in the batch
// (loaded once per tile, reused); each token loads A and B for its adapter_id.
template <int RANK, int TILE_K = 64, int TILE_ROWS = 128>
__global__ void fused_lora_batched_kernel(
    const __half* __restrict__ W,          // [d_out, d_in]
    const __half* __restrict__ A_all,      // [num_adapters, RANK, d_in]
    const __half* __restrict__ B_all,      // [num_adapters, d_out, RANK]
    const __half* __restrict__ X,          // [batch, d_in]
    __half* __restrict__ Y,                // [batch, d_out]
    const int* __restrict__ adapter_ids,   // [batch]
    int batch,
    int d_out,
    int d_in);

// Runtime dispatch by rank.
void launch_fused_lora_batched(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* X,           // [batch, d_in]
    __half* Y,                 // [batch, d_out]
    const int* adapter_ids,    // [batch]
    int batch,
    cudaStream_t stream = nullptr);
