#pragma once
#include <cuda_fp16.h>
#include <cstdint>

struct LoRAConfig {
    int d_in;
    int d_out;
    int rank;
    int num_adapters;
    float alpha;
};

// Memory layout for all adapter weights.
// W:   [d_out, d_in]      fp16, row-major, shared across all adapters
// A:   [num_adapters, rank, d_in]  fp16, each adapter's down-projection
// B:   [num_adapters, d_out, rank] fp16, each adapter's up-projection
struct FusedLoRAWeights {
    const __half* W;  // base weight, size d_out * d_in
    const __half* A;  // all adapter A matrices, size num_adapters * rank * d_in
    const __half* B;  // all adapter B matrices, size num_adapters * d_out * rank
};

__device__ __forceinline__
const __half* adapter_A(const FusedLoRAWeights& w, int adapter_id, int rank, int d_in) {
    return w.A + (int64_t)adapter_id * rank * d_in;
}

__device__ __forceinline__
const __half* adapter_B(const FusedLoRAWeights& w, int adapter_id, int d_out, int rank) {
    return w.B + (int64_t)adapter_id * d_out * rank;
}
