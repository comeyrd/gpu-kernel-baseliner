#include "../NothingWorkload.hpp"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>

#include <vector>

namespace Baseliner {

  __global__ void nothing_kernel(char * /*dummy*/) {
  }

  template <>
  void NothingWorkload<Hardware::CudaBackend>::setup_device(std::shared_ptr<typename backend::stream_t> stream) {
    std::vector<char> host(m_bytes_copied);
    if (m_async_memcpy) {
      CHECK_CUDA(cudaMallocAsync(&m_d_buffer, m_bytes_copied * sizeof(char), *stream));
      CHECK_CUDA(
          cudaMemcpyAsync(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), cudaMemcpyHostToDevice, *stream));
    } else {
      CHECK_CUDA(cudaMalloc(&m_d_buffer, m_bytes_copied * sizeof(char)));
      CHECK_CUDA(cudaMemcpy(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), cudaMemcpyHostToDevice));
    }
  }

  template <>
  void NothingWorkload<Hardware::CudaBackend>::reset_device(std::shared_ptr<typename backend::stream_t> stream) {
    if (m_async_memcpy) {
      CHECK_CUDA(cudaMemsetAsync(m_d_buffer, 0, m_bytes_copied * sizeof(char), *stream));
    } else {
      CHECK_CUDA(cudaMemset(m_d_buffer, 0, m_bytes_copied * sizeof(char)));
    }
  }

  template <>
  auto NothingWorkload<Hardware::CudaBackend>::run(std::shared_ptr<typename backend::stream_t> stream)
      -> std::monostate {
    nothing_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_buffer);
    return {};
  }

  template <>
  void NothingWorkload<Hardware::CudaBackend>::fetch_results(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUDA(cudaFree(m_d_buffer));
    m_d_buffer = nullptr;
  }
  namespace {
    using NothingWorkload = NothingWorkload<Hardware::CudaBackend>;
    BASELINER_REGISTER_WORKLOAD(NothingWorkload);
  } // namespace

} // namespace Baseliner
