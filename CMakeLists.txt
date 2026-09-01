cmake_minimum_required(VERSION 3.18)
project(lora_fuse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

# Static library: fused LoRA kernels
add_library(lora_fuse STATIC
  src/dense_matvec.cu
  src/reference_lora.cu
  src/fused_lora.cu
  src/fused_lora_batched.cu
)

target_include_directories(lora_fuse PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(lora_fuse PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE lora_fuse curand)
target_include_directories(test_correctness PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(test_correctness PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency PRIVATE lora_fuse curand)
target_include_directories(bench_latency PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(bench_latency PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
