#include "reference_lora.cuh"
#include "dense_matvec.cuh"
#include <cuda_fp16.h>

// Kernel: h = A*x where A is [rank, d_in] and x is [d_in].
// Each thread computes one element of h.
__global__ void lora_down_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ x,
    __half* __restrict__ h,
    int rank,
    int d_in)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rank) return;

    float acc = 0.0f;
    const __half* A_row = A + (int64_t)r * d_in;
    for (int k = 0; k < d_in; ++k) {
        acc += __half2float(A_row[k]) * __half2float(x[k]);
    }
    h[r] = __float2half(acc);
}

// Kernel: delta = B*h where B is [d_out, rank] and h is [rank].
// Each thread computes one element of delta.
__global__ void lora_up_kernel(
    const __half* __restrict__ B,
    const __half* __restrict__ h,
    __half* __restrict__ delta,
    int d_out,
    int rank)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d_out) return;

    float acc = 0.0f;
    const __half* B_row = B + (int64_t)row * rank;
    for (int r = 0; r < rank; ++r) {
        acc += __half2float(B_row[r]) * __half2float(h[r]);
    }
    delta[row] = __float2half(acc);
}

// Kernel: y = y_base + alpha * delta, elementwise.
__global__ void lora_add_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ delta,
    float alpha,
    int d_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_out) return;
    float val = __half2float(y[i]) + alpha * __half2float(delta[i]);
    y[i] = __float2half(val);
}

void launch_reference_lora(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* x,
    __half* y,
    __half* h_buf,
    int adapter_id,
    cudaStream_t stream)
{
    const __half* A = adapter_A(weights, adapter_id, cfg.rank, cfg.d_in);
    const __half* B = adapter_B(weights, adapter_id, cfg.d_out, cfg.rank);

    // Step 1: y = W*x (base matvec).
    launch_dense_matvec(weights.W, x, y, cfg.d_out, cfg.d_in, stream);

    // Step 2: h = A*x (lora down-projection).
    {
        constexpr int BLOCK = 64;
        const int grid = (cfg.rank + BLOCK - 1) / BLOCK;
        lora_down_kernel<<<grid, BLOCK, 0, stream>>>(A, x, h_buf, cfg.rank, cfg.d_in);
    }

    // Step 3: delta = B*h (lora up-projection), stored into h_buf reuse via delta ptr.
    // We need a separate delta buffer; reuse the y buffer is not safe.
    // Allocate temporary delta on device using a second provided scratch. For the
    // reference path, we overwrite y in-place by first computing delta to a separate
    // buffer then calling the add kernel. We use dynamic allocation here since this
    // is a reference path (not performance critical).
    __half* delta_buf = nullptr;
    cudaMalloc(&delta_buf, (size_t)cfg.d_out * sizeof(__half));

    {
        constexpr int BLOCK = 128;
        const int grid = (cfg.d_out + BLOCK - 1) / BLOCK;
        lora_up_kernel<<<grid, BLOCK, 0, stream>>>(B, h_buf, delta_buf, cfg.d_out, cfg.rank);
    }

    // Step 4: y += alpha * delta.
    {
        constexpr int BLOCK = 128;
        const int grid = (cfg.d_out + BLOCK - 1) / BLOCK;
        lora_add_kernel<<<grid, BLOCK, 0, stream>>>(y, delta_buf, cfg.alpha, cfg.d_out);
    }

    cudaStreamSynchronize(stream);
    cudaFree(delta_buf);
}
