#include "fused_lora_batched.cuh"
#include <cuda_fp16.h>

// Batched fused kernel.
// Grid: (ceil(d_out / TILE_ROWS), batch)
//   blockIdx.y = token index
//   blockIdx.x = output row tile
//   threadIdx.x = lane within tile (handles one output row)
//
// W is loaded into shared memory identically for every token in the batch.
// A and B are selected per-token from A_all and B_all using adapter_ids[blockIdx.y].
//
// Shared memory layout per block:
//   sx[TILE_K]:       input x tile for this token
//   sA[RANK][TILE_K]: A tile for this token's adapter
template <int RANK, int TILE_K, int TILE_ROWS>
__global__ __launch_bounds__(TILE_ROWS) void fused_lora_batched_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ A_all,
    const __half* __restrict__ B_all,
    const __half* __restrict__ X,
    __half* __restrict__ Y,
    const int* __restrict__ adapter_ids,
    int batch,
    int d_out,
    int d_in)
{
    const int token = blockIdx.y;
    const int row   = blockIdx.x * TILE_ROWS + threadIdx.x;
    if (token >= batch) return;

    const int aid = adapter_ids[token];
    const __half* x = X + (int64_t)token * d_in;
    __half* y = Y + (int64_t)token * d_out;

    // Per-adapter A and B pointers.
    const __half* A = A_all + (int64_t)aid * RANK * d_in;
    const __half* B = B_all + (int64_t)aid * d_out * RANK;

    __shared__ __half sx[TILE_K];
    __shared__ __half sA[RANK][TILE_K];

    float acc_w = 0.0f;
    float acc_a[RANK] = {};

    for (int k_base = 0; k_base < d_in; k_base += TILE_K) {
        const int k_end = min(k_base + TILE_K, d_in);
        const int tile_len = k_end - k_base;

        if (threadIdx.x < tile_len) {
            sx[threadIdx.x] = x[k_base + threadIdx.x];
        }

        for (int load_idx = threadIdx.x; load_idx < RANK * tile_len; load_idx += TILE_ROWS) {
            const int r = load_idx % RANK;
            const int k = load_idx / RANK;
            if (k < tile_len) {
                sA[r][k] = A[(int64_t)r * d_in + k_base + k];
            }
        }
        __syncthreads();

        if (row < d_out) {
            const __half* W_row = W + (int64_t)row * d_in + k_base;
            for (int k = 0; k < tile_len; ++k) {
                acc_w += __half2float(W_row[k]) * __half2float(sx[k]);
            }
            #pragma unroll
            for (int r = 0; r < RANK; ++r) {
                for (int k = 0; k < tile_len; ++k) {
                    acc_a[r] += __half2float(sA[r][k]) * __half2float(sx[k]);
                }
            }
        }
        __syncthreads();
    }

    if (row >= d_out) return;

    // Epilogue: B[row, :] dot acc_a, scaled by alpha.
    // B is loaded directly from global memory into registers (not shared) since
    // each thread needs a different row of B.
    const __half* B_row = B + (int64_t)row * RANK;
    float lora_out = 0.0f;
    // Load alpha from a register via a compile-time cast-safe approach.
    // alpha is passed via a separate device-accessible config. We inline it as
    // a kernel argument rather than a config struct to avoid divergent loads.
    // The batched kernel receives alpha via a uniform __constant__ or per-call
    // parameter. Here we pass via the caller-pushed alpha value.
    // (See launch_fused_lora_batched for how it's forwarded.)
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_out += __half2float(B_row[r]) * acc_a[r];
    }

    y[row] = __float2half(acc_w + lora_out);
    // Note: alpha is baked into B_all at prep time for the batched kernel to avoid
    // an extra per-token uniform load. See launch_fused_lora_batched for scaling.
}

// The batched kernel assumes B has already been scaled by alpha.
// We handle this transparently in the launch function via a temporary scaled copy,
// or by passing alpha as a kernel argument. Here we add alpha as a parameter
// by re-templating with an explicit float arg version.

// Full version with alpha as kernel argument:
template <int RANK, int TILE_K, int TILE_ROWS>
__global__ __launch_bounds__(TILE_ROWS) void fused_lora_batched_alpha_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ A_all,
    const __half* __restrict__ B_all,
    const __half* __restrict__ X,
    __half* __restrict__ Y,
    const int* __restrict__ adapter_ids,
    int batch,
    int d_out,
    int d_in,
    float alpha)
{
    const int token = blockIdx.y;
    const int row   = blockIdx.x * TILE_ROWS + threadIdx.x;
    if (token >= batch) return;

    const int aid = adapter_ids[token];
    const __half* x = X + (int64_t)token * d_in;
    __half* y = Y + (int64_t)token * d_out;

    const __half* A = A_all + (int64_t)aid * RANK * d_in;
    const __half* B = B_all + (int64_t)aid * d_out * RANK;

    __shared__ __half sx[TILE_K];
    __shared__ __half sA[RANK][TILE_K];

    float acc_w = 0.0f;
    float acc_a[RANK] = {};

    for (int k_base = 0; k_base < d_in; k_base += TILE_K) {
        const int k_end = min(k_base + TILE_K, d_in);
        const int tile_len = k_end - k_base;

        if (threadIdx.x < tile_len) {
            sx[threadIdx.x] = x[k_base + threadIdx.x];
        }

        for (int load_idx = threadIdx.x; load_idx < RANK * tile_len; load_idx += TILE_ROWS) {
            const int r = load_idx % RANK;
            const int k = load_idx / RANK;
            if (k < tile_len) {
                sA[r][k] = A[(int64_t)r * d_in + k_base + k];
            }
        }
        __syncthreads();

        if (row < d_out) {
            const __half* W_row = W + (int64_t)row * d_in + k_base;
            for (int k = 0; k < tile_len; ++k) {
                acc_w += __half2float(W_row[k]) * __half2float(sx[k]);
            }
            #pragma unroll
            for (int r = 0; r < RANK; ++r) {
                for (int k = 0; k < tile_len; ++k) {
                    acc_a[r] += __half2float(sA[r][k]) * __half2float(sx[k]);
                }
            }
        }
        __syncthreads();
    }

    if (row >= d_out) return;

    const __half* B_row = B + (int64_t)row * RANK;
    float lora_out = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_out += __half2float(B_row[r]) * acc_a[r];
    }

    y[row] = __float2half(acc_w + alpha * lora_out);
}

// Explicit instantiations.
template __global__ __launch_bounds__(128) void fused_lora_batched_alpha_kernel<8,  128, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*,
    const int*, int, int, int, float);
template __global__ __launch_bounds__(128) void fused_lora_batched_alpha_kernel<16, 128, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*,
    const int*, int, int, int, float);
template __global__ __launch_bounds__(128) void fused_lora_batched_alpha_kernel<32, 64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*,
    const int*, int, int, int, float);
template __global__ __launch_bounds__(128) void fused_lora_batched_alpha_kernel<64, 64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*,
    const int*, int, int, int, float);

void launch_fused_lora_batched(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* X,
    __half* Y,
    const int* adapter_ids,
    int batch,
    cudaStream_t stream)
{
    constexpr int TILE_ROWS = 128;
    const int grid_x = (cfg.d_out + TILE_ROWS - 1) / TILE_ROWS;
    const dim3 grid(grid_x, batch);

    switch (cfg.rank) {
        case 8:
            fused_lora_batched_alpha_kernel<8, 128, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, weights.A, weights.B, X, Y, adapter_ids,
                batch, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 16:
            fused_lora_batched_alpha_kernel<16, 128, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, weights.A, weights.B, X, Y, adapter_ids,
                batch, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 32:
            fused_lora_batched_alpha_kernel<32, 64, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, weights.A, weights.B, X, Y, adapter_ids,
                batch, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 64:
            fused_lora_batched_alpha_kernel<64, 64, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, weights.A, weights.B, X, Y, adapter_ids,
                batch, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        default:
            fused_lora_batched_alpha_kernel<64, 64, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, weights.A, weights.B, X, Y, adapter_ids,
                batch, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
    }
}
