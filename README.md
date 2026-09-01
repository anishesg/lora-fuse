# lora-fuse

Fused LoRA decode inference: base weight matvec with register-resident low-rank accumulation and multi-adapter batched dispatch.

## Problem

LoRA fine-tuning decomposes a weight update as delta_W = B * A, where A has shape (rank x d_in) and B has shape (d_out x rank) with rank << d_in, d_out. During decode inference for a single token, the forward pass for one linear layer requires:

1. y = W * x (base matvec, shape d_out)
2. h = A * x (low-rank down-projection, shape r)
3. delta = B * h (low-rank up-projection, shape d_out)
4. y += alpha * delta (elementwise add)

Naive approaches (S-LoRA BGMV, Punica segmented gather) schedule these as separate CUDA kernel launches. The cost per linear layer at decode time:

- 2-3 extra kernel launches (kernel launch latency ~5-10 us each)
- Global memory round-trip for the rank-r intermediate h: write r float16 values after step 2, read them back in step 3
- W, A, B each loaded from global memory in separate passes despite x being the same input

At batch size 1 (single-token decode), each linear layer in a 70B LLaMA-class model has ~80 linear projections across layers. The overhead from separate kernels compounds quickly in a latency-sensitive serving path.

## Fusion Insight

The rank-r intermediate vector h = A*x is tiny: 8 to 64 float32 values. This fits in registers on any modern GPU warp. Instead of writing h to global memory between kernel 1 and kernel 2, we can accumulate it during the base matvec pass:

In a standard tiled W*x matvec, each tile iteration:
- Loads TILE_K columns of W into shared memory
- Loads the corresponding TILE_K elements of x into shared memory
- Accumulates partial W*x sums in fp32 registers

With LoRA fusion, the same tile iteration additionally:
- Loads the corresponding TILE_K columns of A (r x TILE_K) into registers
- Accumulates partial A*x sums in r fp32 registers simultaneously

After all tiles complete, each thread holds both its partial W*x sum and the full A*x vector in registers. The B*(A*x) epilogue then loads B (d_out x r) from shared memory and computes the rank-r dot product from the register accumulator, adding alpha * result to the W*x accumulator before writing the final fp16 output.

**Net result**: base + LoRA output in a single kernel launch, with zero extra global memory traffic for the low-rank intermediate.

## Comparison with Prior Work

| Approach | Kernel launches | Intermediate memory | Multi-adapter |
|---|---|---|---|
| Naive separate | 3 (W*x, A*x, B*h) | r x sizeof(fp16) global | N adapters = 3N launches |
| S-LoRA BGMV | 2 (base + fused BGMV) | none | segmented-gather, variable padding |
| Punica | 2 (base + BGMV) | none | batched across segments |
| **lora-fuse** | **1** | **none** | **per-token adapter_id, single launch** |

S-LoRA BGMV and Punica fuse B*h and optionally combine with base, but still separate the A*x pass. lora-fuse goes one step further: the A accumulation is folded into the tile loop of the base W*x pass, so A is never materialized and no extra launch occurs.

For multi-tenant serving (different requests using different LoRA adapters), lora-fuse accepts a per-token adapter_ids array. All tokens share the same W load (one pass through global memory for the base weight), but each token independently selects its A and B matrices by adapter_id. This eliminates per-adapter dispatch overhead entirely.

## Architecture

```
src/
  lora_config.cuh          # LoRAConfig struct, FusedLoRAWeights layout, device accessors
  dense_matvec.cuh/.cu     # Tiled base W*x kernel (baseline, no LoRA)
  reference_lora.cuh/.cu   # Two-kernel reference: separate A*x and B*h (correctness oracle)
  fused_lora.cuh/.cu       # Single-kernel fused base+LoRA, template-specialized for r in {8,16,32,64}
  fused_lora_batched.cuh/.cu # Multi-adapter batched kernel with per-token adapter_ids
tests/
  test_correctness.cu      # Cosine similarity and max absolute error vs reference
benchmarks/
  bench_latency.cu         # CUDA event timing, bandwidth analysis, tokens/sec throughput
```

## Build

Requires CUDA 11.8+ and a GPU with compute capability 8.0+ (Ampere or newer).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_latency
```

To target a specific architecture:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
```

## Key Parameters

- **TILE_K**: columns of W processed per tile iteration (tuned per kernel)
- **TILE_ROWS**: rows of W assigned per thread block
- **rank r**: compile-time template parameter for A*x register accumulator size
- **alpha**: LoRA scaling factor (typically 1/r or a fixed scalar)
