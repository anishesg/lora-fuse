#pragma once
#include <cuda_fp16.h>

// Tiled W*x matvec, no LoRA.
// W: [d_out, d_in] fp16 row-major
// x: [d_in] fp16
// y: [d_out] fp16
// Template params:
//   TILE_K:    columns of W loaded per tile (must divide d_in or kernel handles tail)
//   TILE_ROWS: rows of W assigned per thread block
template <int TILE_K = 64, int TILE_ROWS = 128>
__global__ void dense_matvec_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int d_out,
    int d_in);

void launch_dense_matvec(
    const __half* W,
    const __half* x,
    __half* y,
    int d_out,
    int d_in,
    cudaStream_t stream = nullptr);
