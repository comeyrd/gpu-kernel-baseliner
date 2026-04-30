#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

__global__ void warmup_kernel(float *__restrict__ data, size_t n, int iterations) {
  size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float x = data[idx];

  for (int i = 0; i < iterations; ++i) {
    // Heavy FMA chain - maximizes ALU heat
    x = fmaf(x, 1.00001f, 0.00001f);
    x = fmaf(x, x, 0.00001f);
    x = fmaf(x, 1.00001f, x);
    x = fmaf(x, x, 0.00001f);
    x = fmaf(x, 1.00001f, 0.00001f);
    x = fmaf(x, x, 0.00001f);
    x = fmaf(x, 1.00001f, x);
    x = fmaf(x, x, 0.00001f);

    // Periodic write-back to force memory traffic
    if (i % 64 == 0) {
      data[idx] = x;
      // Also hammer a neighbor to thrash cache lines
      size_t neighbor = (idx + blockDim.x) % n;
      data[neighbor] = x;
      x = data[(idx + n / 2) % n]; // cross-read to stress DRAM BW
    }
  }

  data[idx] = x;
}

namespace Baseliner::Hardware {

  template <>
  void CudaBackend::warm_gpu(stream_t &stream) {
    constexpr int ITERATIONS = 5; // Tune to desired warm-up duration
    constexpr int BLOCK_SIZE = 256;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t alloc_bytes = static_cast<size_t>(free_bytes * 0.50);
    const size_t N = alloc_bytes / sizeof(float);
    const size_t GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(float)));
    warmup_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(d_data, N, ITERATIONS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_data));
  }

} // namespace Baseliner::Hardware