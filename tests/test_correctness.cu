#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include <curand.h>

#include "lora_config.cuh"
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

__global__ void scale_fp32_to_fp16(const float* src, __half* dst, size_t n, float scale, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int)n) dst[i] = __float2half(src[i] * scale + offset);
}

// Fill device buffer with uniform random fp16 in [-0.1, 0.1].
static void rand_fp16(curandGenerator_t gen, __half* d_buf, size_t n) {
    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));
    CURAND_CHECK(curandGenerateUniform(gen, d_tmp, n));
    int grid = (int)((n + 255) / 256);
    scale_fp32_to_fp16<<<grid, 256>>>(d_tmp, d_buf, n, 0.2f, -0.1f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_tmp));
}

static std::vector<__half> to_host(const __half* d, size_t n) {
    std::vector<__half> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost));
    return h;
}

static float cosine_sim(const std::vector<__half>& a, const std::vector<__half>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float ai = __half2float(a[i]);
        float bi = __half2float(b[i]);
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    if (na == 0 || nb == 0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float max_abs_err(const std::vector<__half>& a, const std::vector<__half>& b) {
    float mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (diff > mx) mx = diff;
    }
    return mx;
}

struct TestResult {
    bool passed;
    float cosine;
    float max_err;
};

static TestResult test_fused_single(
    curandGenerator_t gen,
    int d_in, int d_out, int rank, float alpha)
{
    LoRAConfig cfg{d_in, d_out, rank, 1, alpha};

    size_t W_sz  = (size_t)d_out * d_in;
    size_t A_sz  = (size_t)rank  * d_in;
    size_t B_sz  = (size_t)d_out * rank;

    __half *d_W, *d_A, *d_B, *d_x, *d_y_ref, *d_y_fused, *d_h;
    CUDA_CHECK(cudaMalloc(&d_W,       W_sz  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A,       A_sz  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B,       B_sz  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_x,       d_in  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_ref,   d_out * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_y_fused, d_out * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h,       rank  * sizeof(__half)));

    rand_fp16(gen, d_W, W_sz);
    rand_fp16(gen, d_A, A_sz);
    rand_fp16(gen, d_B, B_sz);
    rand_fp16(gen, d_x, d_in);

    FusedLoRAWeights weights{d_W, d_A, d_B};

    launch_reference_lora(weights, cfg, d_x, d_y_ref, d_h, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_fused_lora(weights, cfg, d_x, d_y_fused, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ref   = to_host(d_y_ref,   d_out);
    auto fused = to_host(d_y_fused, d_out);

    float cos = cosine_sim(ref, fused);
    float mae = max_abs_err(ref, fused);
    bool ok = (cos > 0.999f);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y_ref)); CUDA_CHECK(cudaFree(d_y_fused));
    CUDA_CHECK(cudaFree(d_h));

    return {ok, cos, mae};
}

static TestResult test_fused_batched(
    curandGenerator_t gen,
    int d_in, int d_out, int rank, float alpha,
    int batch, int num_adapters)
{
    LoRAConfig cfg{d_in, d_out, rank, num_adapters, alpha};

    size_t W_sz     = (size_t)d_out * d_in;
    size_t A_all_sz = (size_t)num_adapters * rank * d_in;
    size_t B_all_sz = (size_t)num_adapters * d_out * rank;
    size_t X_sz     = (size_t)batch * d_in;
    size_t Y_sz     = (size_t)batch * d_out;
    size_t h_sz     = (size_t)rank;

    __half *d_W, *d_A_all, *d_B_all, *d_X, *d_Y_batched, *d_Y_ref, *d_h;
    int* d_adapter_ids;

    CUDA_CHECK(cudaMalloc(&d_W,         W_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_A_all,     A_all_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B_all,     B_all_sz * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_X,         X_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Y_batched, Y_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_Y_ref,     Y_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_h,         h_sz     * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_adapter_ids, batch  * sizeof(int)));

    rand_fp16(gen, d_W,     W_sz);
    rand_fp16(gen, d_A_all, A_all_sz);
    rand_fp16(gen, d_B_all, B_all_sz);
    rand_fp16(gen, d_X,     X_sz);

    // Build adapter_ids on host: cycle through [0, num_adapters).
    std::vector<int> h_ids(batch);
    for (int i = 0; i < batch; ++i) h_ids[i] = i % num_adapters;
    CUDA_CHECK(cudaMemcpy(d_adapter_ids, h_ids.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

    FusedLoRAWeights weights{d_W, d_A_all, d_B_all};

    // Run batched fused kernel.
    launch_fused_lora_batched(weights, cfg, d_X, d_Y_batched, d_adapter_ids, batch, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Build reference: per-token calls.
    LoRAConfig single_cfg{d_in, d_out, rank, num_adapters, alpha};
    for (int t = 0; t < batch; ++t) {
        const __half* x_t = d_X + (int64_t)t * d_in;
        __half* y_t = d_Y_ref + (int64_t)t * d_out;
        launch_reference_lora(weights, single_cfg, x_t, y_t, d_h, h_ids[t], nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ref     = to_host(d_Y_ref,     Y_sz);
    auto batched = to_host(d_Y_batched, Y_sz);

    float cos = cosine_sim(ref, batched);
    float mae = max_abs_err(ref, batched);
    bool ok = (cos > 0.999f);

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_A_all)); CUDA_CHECK(cudaFree(d_B_all));
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Y_batched)); CUDA_CHECK(cudaFree(d_Y_ref));
    CUDA_CHECK(cudaFree(d_h)); CUDA_CHECK(cudaFree(d_adapter_ids));

    return {ok, cos, mae};
}

int main() {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));

    int total = 0, passed = 0;
    bool any_fail = false;

    printf("%-10s %-10s %-6s %-6s %-12s %-12s %-8s\n",
           "d_in", "d_out", "rank", "alpha", "cosine", "max_err", "status");
    printf("%s\n", std::string(72, '-').c_str());

    // Single-token fused kernel tests.
    const int d_vals[]    = {4096, 8192};
    const int rank_vals[] = {8, 16, 32, 64};
    const float alpha_vals[] = {0.5f, 1.0f};

    for (int d_in : d_vals) {
        for (int d_out : d_vals) {
            for (int rank : rank_vals) {
                for (float alpha : alpha_vals) {
                    auto r = test_fused_single(gen, d_in, d_out, rank, alpha);
                    ++total;
                    if (r.passed) ++passed; else any_fail = true;
                    printf("%-10d %-10d %-6d %-6.1f %-12.6f %-12.6f %s\n",
                           d_in, d_out, rank, alpha, r.cosine, r.max_err,
                           r.passed ? "PASS" : "FAIL");
                }
            }
        }
    }

    printf("\n--- Batched multi-adapter tests (batch=16, num_adapters=4) ---\n");
    printf("%-10s %-10s %-6s %-6s %-12s %-12s %-8s\n",
           "d_in", "d_out", "rank", "alpha", "cosine", "max_err", "status");
    printf("%s\n", std::string(72, '-').c_str());

    for (int rank : rank_vals) {
        for (float alpha : alpha_vals) {
            auto r = test_fused_batched(gen, 4096, 4096, rank, alpha, 16, 4);
            ++total;
            if (r.passed) ++passed; else any_fail = true;
            printf("%-10d %-10d %-6d %-6.1f %-12.6f %-12.6f %s\n",
                   4096, 4096, rank, alpha, r.cosine, r.max_err,
                   r.passed ? "PASS" : "FAIL");
        }
    }

    printf("\n%d / %d tests passed.\n", passed, total);

    CURAND_CHECK(curandDestroyGenerator(gen));
    return any_fail ? 1 : 0;
}
