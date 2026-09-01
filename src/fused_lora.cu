#include "fused_lora.cuh"
#include <cuda_fp16.h>

// Fused kernel implementation.
//
// Each thread handles one output row (row = blockIdx.x * TILE_ROWS + threadIdx.x).
// Per tile iteration:
//   - shared: sx[TILE_K] holds tile of input x
//   - shared: sA[RANK][TILE_K] holds tile of A rows (all ranks, same k columns as W tile)
//   - thread-local: acc_w accumulates W[row, :] dot x
//   - thread-local: acc_a[RANK] accumulates A[:, :] dot x (all r rows, per thread)
//
// After tile loop, epilogue:
//   - Load B[row, :] (shape [RANK]) from global into registers
//   - Compute dot(B_row, acc_a) -> scalar
//   - y[row] = fp16(acc_w + alpha * dot(B_row, acc_a))
template <int RANK, int TILE_K, int TILE_ROWS>
__global__ void fused_lora_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int d_out,
    int d_in,
    float alpha)
{
    const int row = blockIdx.x * TILE_ROWS + threadIdx.x;

    __shared__ __half sx[TILE_K];
    // sA[RANK][TILE_K]: each of RANK rows of A for this tile
    __shared__ __half sA[RANK][TILE_K];

    float acc_w = 0.0f;
    float acc_a[RANK] = {};  // compiler unrolls for compile-time RANK

    for (int k_base = 0; k_base < d_in; k_base += TILE_K) {
        const int k_end = min(k_base + TILE_K, d_in);
        const int tile_len = k_end - k_base;

        // Load x tile (first TILE_K threads load, all use).
        if (threadIdx.x < tile_len) {
            sx[threadIdx.x] = x[k_base + threadIdx.x];
        }

        // Load A tile: RANK rows x tile_len cols.
        // Distribute across threads: thread t loads sA[t % RANK][t / RANK ... ] column-strided.
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
            // Accumulate W*x partial sum.
            for (int k = 0; k < tile_len; ++k) {
                acc_w += __half2float(W_row[k]) * __half2float(sx[k]);
            }
            // Accumulate A*x partial sums for all ranks.
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

    // Epilogue: compute alpha * B[row, :] dot acc_a.
    const __half* B_row = B + (int64_t)row * RANK;
    float lora_out = 0.0f;
    #pragma unroll
    for (int r = 0; r < RANK; ++r) {
        lora_out += __half2float(B_row[r]) * acc_a[r];
    }

    y[row] = __float2half(acc_w + alpha * lora_out);
}

// Template specializations for rank 8, 16, 32, 64.
// Explicit instantiations ensure the compiler allocates exactly RANK registers
// for acc_a and fully unrolls the epilogue and inner A*x loops.

template __global__ void fused_lora_kernel<8,  64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_lora_kernel<16, 64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_lora_kernel<32, 64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);
template __global__ void fused_lora_kernel<64, 64, 128>(
    const __half*, const __half*, const __half*, const __half*, __half*, int, int, float);

void launch_fused_lora(
    const FusedLoRAWeights& weights,
    const LoRAConfig& cfg,
    const __half* x,
    __half* y,
    int adapter_id,
    cudaStream_t stream)
{
    constexpr int TILE_K = 64;
    constexpr int TILE_ROWS = 128;
    const int grid = (cfg.d_out + TILE_ROWS - 1) / TILE_ROWS;

    const __half* A = adapter_A(weights, adapter_id, cfg.rank, cfg.d_in);
    const __half* B = adapter_B(weights, adapter_id, cfg.d_out, cfg.rank);

    switch (cfg.rank) {
        case 8:
            fused_lora_kernel<8,  TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, A, B, x, y, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 16:
            fused_lora_kernel<16, TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, A, B, x, y, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 32:
            fused_lora_kernel<32, TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, A, B, x, y, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        case 64:
            fused_lora_kernel<64, TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, A, B, x, y, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
        default:
            // Fallback: treat as rank 64 with masking (never triggered for valid configs)
            fused_lora_kernel<64, TILE_K, TILE_ROWS><<<grid, TILE_ROWS, 0, stream>>>(
                weights.W, A, B, x, y, cfg.d_out, cfg.d_in, cfg.alpha);
            break;
    }
}
