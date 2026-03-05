#include "hip/hip_runtime.h"
#include "../NothingKernel.hpp"
#include "baseliner/managers/RegisteringMacros.hpp"
#include <baseliner/hardware/hip/HipBackend.hpp>
using namespace Baseliner::Hardware;
namespace Baseliner {
  __global__ void nothing_kernel(char *dummy) {
  }
  template <>
  void NothingKernel<HipBackend>::setup(std::shared_ptr<HipBackend::stream_t> stream) {
    std::vector<char> host;
    host.reserve(m_bytes_copied);
    if (m_async_memcpy) {
      CHECK_HIP(hipMallocAsync(&m_d_buffer, m_bytes_copied * sizeof(char), *stream));
      CHECK_HIP(
          hipMemcpyAsync(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), hipMemcpyHostToDevice, *stream));
    } else {
      CHECK_HIP(hipMalloc(&m_d_buffer, m_bytes_copied * sizeof(char)));
      CHECK_HIP(hipMemcpy(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), hipMemcpyHostToDevice));
    }
  };
  template <>
  void NothingKernel<HipBackend>::reset_case(std::shared_ptr<typename backend::stream_t> stream) {
    if (m_async_memcpy) {
      CHECK_HIP(hipMemsetAsync(m_d_buffer, 0, m_bytes_copied * sizeof(char), *stream));
    } else {
      CHECK_HIP(hipMemset(m_d_buffer, 0, m_bytes_copied * sizeof(char)));
    }
  };
  template <>
  void NothingKernel<HipBackend>::run_case(std::shared_ptr<typename backend::stream_t> stream) {
    nothing_kernel<<<m_blocks, m_threads, 0, *stream>>>(m_d_buffer);
  };
  template <>
  void NothingKernel<HipBackend>::teardown(std::shared_ptr<typename backend::stream_t> stream) {
    CHECK_HIP(hipFree(m_d_buffer));
  };
  namespace {
    using NothingKernel = NothingKernel<HipBackend>;
    BASELINER_REGISTER_CASE(NothingKernel);
  } // namespace
} // namespace Baseliner
