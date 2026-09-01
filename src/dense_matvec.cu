#include "dense_matvec.cuh"
#include <cuda_fp16.h>

template <int TILE_K, int TILE_ROWS>
__global__ void dense_matvec_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int d_out,
    int d_in)
{
    // Each thread block handles TILE_ROWS output rows.
    // Each thread handles one output row within the block.
    const int row = blockIdx.x * TILE_ROWS + threadIdx.x;
    if (row >= d_out) return;

    __shared__ __half sx[TILE_K];  // tile of input x

    float acc = 0.0f;

    for (int k_base = 0; k_base < d_in; k_base += TILE_K) {
        const int k_end = min(k_base + TILE_K, d_in);
        const int tile_len = k_end - k_base;

        // Cooperatively load x tile into shared memory.
        if (threadIdx.x < tile_len) {
            sx[threadIdx.x] = x[k_base + threadIdx.x];
        }
        __syncthreads();

        // Accumulate W[row, k_base:k_end] dot sx.
        const __half* W_row = W + (int64_t)row * d_in + k_base;
        for (int k = 0; k < tile_len; ++k) {
            acc += __half2float(W_row[k]) * __half2float(sx[k]);
        }
        __syncthreads();
    }

    y[row] = __float2half(acc);
}

void launch_dense_matvec(
    const __half* W,
    const __half* x,
    __half* y,
    int d_out,
    int d_in,
    cudaStream_t stream)
{
    constexpr int TILE_K = 64;
    constexpr int TILE_ROWS = 128;
    const int grid = (d_out + TILE_ROWS - 1) / TILE_ROWS;
    dense_matvec_kernel<TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
        W, x, y, d_out, d_in);
}
