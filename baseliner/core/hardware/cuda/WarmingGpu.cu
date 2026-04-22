#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

__global__ void warm_kernel(float *data, unsigned long n, int iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float val = data[idx];
  for (int i = 0; i < iterations; ++i) {
    val = val * 1.0001f + 0.0001f;
    val = val + 1.23512f + 2.3215f;
  }
  data[idx] = val;
  __syncthreads();
  for (int i = 0; i < iterations; ++i) {
    val = val * 1.0001f + 0.0001f;
    val = val + 1.23512f + 2.3215f;
  }
  if (idx != 0) {
    data[idx - 1] = data[idx] + val * val;
  }
}

namespace Baseliner::Hardware {

  template <>
  void CudaBackend::warm_gpu(std::shared_ptr<stream_t> stream) {
    constexpr size_t N = 10000000;
    constexpr int ITERATIONS = 2000; // Tune to desired warm-up duration
    constexpr int BLOCK_SIZE = 256;
    constexpr size_t GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_data, 0, N * sizeof(float)));

    warm_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, *stream>>>(d_data, N, ITERATIONS);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(*stream));
    CHECK_CUDA(cudaFree(d_data));
  }

} // namespace Baseliner::Hardware