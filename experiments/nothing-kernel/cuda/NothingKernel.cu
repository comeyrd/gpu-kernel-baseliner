#include "../NothingKernel.hpp"
#include "baseliner/managers/RegisteringMacros.hpp"
#include <baseliner/hardware/cuda/CudaBackend.hpp>
using namespace Baseliner::Hardware;
namespace Baseliner {
  __global__ void nothing_kernel(char *dummy) {
  }
  template <>
  void NothingKernel<CudaBackend>::setup(std::shared_ptr<CudaBackend::stream_t> stream) {
    std::vector<char> host;
    host.reserve(m_bytes_copied);
    if (m_async_memcpy) {
      CHECK_CUDA(cudaMallocAsync(&m_d_buffer, m_bytes_copied * sizeof(char), *stream));
      CHECK_CUDA(
          cudaMemcpyAsync(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), cudaMemcpyHostToDevice, *stream));
    } else {
      CHECK_CUDA(cudaMalloc(&m_d_buffer, m_bytes_copied * sizeof(char)));
      CHECK_CUDA(cudaMemcpy(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), cudaMemcpyHostToDevice));
    }
  };
  template <>
  void NothingKernel<CudaBackend>::reset_case(std::shared_ptr<typename backend::stream_t> stream) {
    if (m_async_memcpy) {
      CHECK_CUDA(cudaMemsetAsync(m_d_buffer, 0, m_bytes_copied * sizeof(char), *stream));
    } else {
      CHECK_CUDA(cudaMemset(m_d_buffer, 0, m_bytes_copied * sizeof(char)));
    }
  };
  template <>
  void NothingKernel<CudaBackend>::run_case(std::shared_ptr<typename backend::stream_t> stream) {
    nothing_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_buffer);
  };
  template <>
  void NothingKernel<CudaBackend>::teardown(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_CUDA(cudaFree(m_d_buffer));
  };
  namespace {
    using NothingKernel = NothingKernel<CudaBackend>;
    BASELINER_REGISTER_CASE(NothingKernel);
  } // namespace
} // namespace Baseliner