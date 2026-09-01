#pragma once
#include <cuda_fp16.h>
#include "lora_config.cuh"

// Two-kernel reference LoRA path used as correctness oracle.
// Computes: y = W*x + alpha * B*(A*x)
// Steps:
//   1. dense_matvec: y_base = W*x
//   2. lora_down:    h = A*x  (stored in global memory, shape [rank])
//   3. lora_up:      delta = B*h  (shape [d_out])
//   4. elementwise:  y = y_base + alpha * delta
//
// h_buf must be pre-allocated with at least rank float16 elements.
void launch_reference_lora(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* x,       // [d_in]
    __half* y,             // [d_out] output
    __half* h_buf,         // [rank] scratch buffer for A*x intermediate
    int adapter_id,
    cudaStream_t stream = nullptr);
