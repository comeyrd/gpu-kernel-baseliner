#include "hip/hip_runtime.h"
#include <baseliner/core/hardware/hip/HipBackend.hpp>
// WarmingKernel.hip or HipBackend.cpp

__global__ void warm_kernel(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float x = 0.5f + idx * 0.0001f;
  for (int i = 0; i < 10000; ++i) {
    x += sinf(x) * cosf(x);
    x *= 1.0000001f;
    x = sqrtf(x + 1.0f);
    x = logf(x + 1.0f);
  }
  data[idx] = x;
}
namespace Baseliner::Hardware {
  template <>
  void WarmingKernel<HipBackend>::alloc(HipBackend::stream_t stream) {
    int grid_size = 0;
    int threads_per_block = 0;
    CHECK_HIP(hipOccupancyMaxPotentialBlockSize(&grid_size, &threads_per_block, warm_kernel, 0, 0));
    m_num_items = grid_size * threads_per_block;
    m_threads_per_block = threads_per_block;
    CHECK_HIP(hipMalloc(&m_data, m_num_items * sizeof(float)));
  }

  template <>
  void WarmingKernel<HipBackend>::free() {
    if (m_data) {
      CHECK_HIP(hipFree(m_data));
      m_data = nullptr;
      m_num_items = 0;
    }
  }

  template <>
  void WarmingKernel<HipBackend>::warm(HipBackend::stream_t stream) {
    int grid_size = m_num_items / m_threads_per_block;
    warm_kernel<<<dim3(grid_size), dim3(m_threads_per_block), 0, stream>>>(m_data, m_num_items);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipStreamSynchronize(stream));
  }
} // namespace Baseliner::Hardware