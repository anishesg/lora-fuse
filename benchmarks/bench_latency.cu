#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <vector>
#include <cuda_fp16.h>
#include <curand.h>

#include "lora_config.cuh"
#include "dense_matvec.cuh"
#include "reference_lora.cuh"
#include "fused_lora.cuh"
#include "fused_lora_batched.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define CURAND_CHECK(call) do { \
    curandStatus_t err = (call); \
    if (err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while (0)

__global__ void scale_f32_to_f16_bench(const float* src, __half* dst, size_t n, float scale, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] = __float2half(src[i] * scale + offset);
}

static void rand_fp16(curandGenerator_t gen, __half* d, size_t n) {
    float* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
    CURAND_CHECK(curandGenerateUniform(gen, tmp, n));
    scale_f32_to_f16_bench<<<(int)((n+255)/256), 256>>>(tmp, d, n, 0.2f, -0.1f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(tmp));
}

// Returns mean latency in microseconds over REPS warmup-corrected runs.
static float time_kernel(int reps, std::function<void()> fn) {
    // Warmup.
    for (int i = 0; i < 3; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (ms / reps) * 1000.0f;  // microseconds
}

static void bench_single_token(curandGenerator_t gen, int d_in, int d_out, int rank) {
    const float alpha = 1.0f;
    LoRAConfig cfg{d_in, d_out, rank, 1, alpha};

    size_t W_sz = (size_t)d_out * d_in;
    size_t A_sz = (size_t)rank  * d_in;
    size_t B_sz = (size_t)d_out * rank;

    __half *d_W, *d_A, *d_B, *d_x, *d_y, *d_h;
    CUDA_CHECK(cudaMalloc(&d_W, W_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A, A_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, B_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x, d_in  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y, d_out * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h, rank  * sizeof(__half)));

    rand_fp16(gen, d_W, W_sz);
    rand_fp16(gen, d_A, A_sz);
    rand_fp16(gen, d_B, B_sz);
    rand_fp16(gen, d_x, d_in);

    FusedLoRAWeights weights{d_W, d_A, d_B};

    constexpr int REPS = 100;

    float t_dense = time_kernel(REPS, [&]() {
        launch_dense_matvec(d_W, d_x, d_y, d_out, d_in, nullptr);
    });

    float t_ref = time_kernel(REPS, [&]() {
        launch_reference_lora(weights, cfg, d_x, d_y, d_h, 0, nullptr);
    });

    float t_fused = time_kernel(REPS, [&]() {
        launch_fused_lora(weights, cfg, d_x, d_y, 0, nullptr);
    });

    // Bytes loaded analysis (ideal, assuming no caching):
    //   dense:     W bytes = d_out * d_in * 2  +  x bytes = d_in * 2
    //   reference: W + x  (W*x)  +  A + x  (A*x)  +  B + h  (B*h)
    //              = 2*(W+x) + A + B + rank*2 + A*x intermediate
    //   fused:     W + x  +  A (during W tile loop)  +  B (epilogue)
    //              = W + A + B + x  (x loaded once, not twice)
    double bytes_W = (double)d_out * d_in * 2;
    double bytes_A = (double)rank  * d_in * 2;
    double bytes_B = (double)d_out * rank * 2;
    double bytes_x = (double)d_in  * 2;
    double bytes_h = (double)rank  * 2;

    double bw_dense = (bytes_W + bytes_x) / (t_dense * 1e-6) / 1e9;  // GB/s
    double bw_ref   = (2*(bytes_W+bytes_x) + bytes_A + bytes_B + bytes_h) / (t_ref * 1e-6) / 1e9;
    double bw_fused = (bytes_W + bytes_A + bytes_B + bytes_x) / (t_fused * 1e-6) / 1e9;

    printf("  d_in=%-5d d_out=%-5d rank=%-3d | dense=%7.2f us | ref=%7.2f us | fused=%7.2f us | "
           "speedup=%.2fx | bw_fused=%.1f GB/s\n",
           d_in, d_out, rank, t_dense, t_ref, t_fused, t_ref / t_fused, bw_fused);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_h));
}

static void bench_batched(curandGenerator_t gen, int d_in, int d_out, int rank,
                           int batch, int num_adapters) {
    const float alpha = 1.0f;
    LoRAConfig cfg{d_in, d_out, rank, num_adapters, alpha};

    size_t W_sz     = (size_t)d_out * d_in;
    size_t A_all_sz = (size_t)num_adapters * rank * d_in;
    size_t B_all_sz = (size_t)num_adapters * d_out * rank;
    size_t X_sz     = (size_t)batch * d_in;
    size_t Y_sz     = (size_t)batch * d_out;

    __half *d_W, *d_A_all, *d_B_all, *d_X, *d_Y;
    int* d_ids;
    CUDA_CHECK(cudaMalloc(&d_W,     W_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A_all, A_all_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_all, B_all_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_X,     X_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Y,     Y_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ids,   batch    * sizeof(int)));

    rand_fp16(gen, d_W,     W_sz);
    rand_fp16(gen, d_A_all, A_all_sz);
    rand_fp16(gen, d_B_all, B_all_sz);
    rand_fp16(gen, d_X,     X_sz);

    std::vector<int> h_ids(batch);
    for (int i = 0; i < batch; ++i) h_ids[i] = i % num_adapters;
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

    FusedLoRAWeights weights{d_W, d_A_all, d_B_all};

    constexpr int REPS = 100;
    float t_batched = time_kernel(REPS, [&]() {
        launch_fused_lora_batched(weights, cfg, d_X, d_Y, d_ids, batch, nullptr);
    });

    double throughput = (double)batch / (t_batched * 1e-6);  // tokens/sec

    printf("  d_in=%-5d d_out=%-5d rank=%-3d batch=%-3d adapters=%-3d | "
           "latency=%7.2f us | throughput=%.0f tokens/sec\n",
           d_in, d_out, rank, batch, num_adapters, t_batched, throughput);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_A_all)); CUDA_CHECK(cudaFree(d_B_all));
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Y)); CUDA_CHECK(cudaFree(d_ids));
}

int main() {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 99ULL));

    // Print device info.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d)  Peak BW: %.0f GB/s\n\n",
           prop.name, prop.major, prop.minor,
           (double)prop.memoryBusWidth / 8 * prop.memoryClockRate * 2 * 1e-6);

    printf("=== Single-token: dense vs reference (3-kernel) vs fused ===\n");
    const int d_vals[] = {4096, 8192};
    const int rank_vals[] = {8, 16, 32, 64};
    for (int d_in : d_vals) {
        for (int d_out : d_vals) {
            for (int rank : rank_vals) {
                bench_single_token(gen, d_in, d_out, rank);
            }
        }
    }

    printf("\n=== Multi-adapter batched: varying batch size and num_adapters ===\n");
    const int batch_sizes[]    = {1, 4, 8, 16, 32};
    const int adapter_counts[] = {1, 4, 16};
    for (int batch : batch_sizes) {
        for (int na : adapter_counts) {
            if (na > batch) continue;  // num_adapters <= batch is meaningful
            bench_batched(gen, 4096, 4096, 32, batch, na);
        }
    }

    CURAND_CHECK(curandDestroyGenerator(gen));
    return 0;
}
