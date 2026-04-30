#include "hip/hip_runtime.h"

#include "../ComputationWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#include <vector>

__global__ void computation_kernel(int *a, int *b, int *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
  __syncthreads();
  for (int i = 0; i < 50; i++) {
    if (idx > 0 && idx < N) {
      c[idx] = a[idx] + c[idx] + b[idx] * b[idx];
    }
    __syncthreads();
  }
}
template <>
auto ComputationWorkload<Baseliner::Hardware::HipBackend>::run(std::shared_ptr<hipStream_t> stream) -> std::monostate {
  computation_kernel<<<m_blocksPerGrid, m_threadsPerBlock, 0, *stream>>>(m_d_a, m_d_b, m_d_c, m_N);
  return {};
}

template <>
void ComputationWorkload<Baseliner::Hardware::HipBackend>::setup_device(
    std::shared_ptr<typename backend::stream_t> stream) {
  CHECK_HIP(hipMallocAsync(&m_d_a, m_N * sizeof(int), *stream));
  CHECK_HIP(hipMallocAsync(&m_d_b, m_N * sizeof(int), *stream));
  CHECK_HIP(hipMallocAsync(&m_d_c, m_N * sizeof(int), *stream));

  m_threadsPerBlock = 256;
  m_blocksPerGrid = (m_N + m_threadsPerBlock - 1) / m_threadsPerBlock;

  CHECK_HIP(hipMemcpy(m_d_a, m_a_host.data(), m_N * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(m_d_b, m_b_host.data(), m_N * sizeof(int), hipMemcpyHostToDevice));
}

template <typename BackendT>
void ComputationWorkload<BackendT>::reset_device(std::shared_ptr<typename backend::stream_t> stream) {
}

template <typename BackendT>
void ComputationWorkload<BackendT>::fetch_results(std::shared_ptr<typename backend::stream_t> stream) {
  CHECK_HIP(hipMemcpyAsync(m_c_host.data(), m_d_c, m_N * sizeof(int), hipMemcpyDeviceToHost, *stream));
  CHECK_HIP(hipStreamSynchronize(*stream));
  CHECK_HIP(hipFree(m_d_a));
  CHECK_HIP(hipFree(m_d_b));
  CHECK_HIP(hipFree(m_d_c));
  m_d_a = nullptr;
  m_d_b = nullptr;
  m_d_c = nullptr;
}
namespace {
  using ComputationWorkload = ComputationWorkload<Baseliner::Hardware::HipBackend>;
  BASELINER_REGISTER_WORKLOAD(ComputationWorkload);
} // namespace
