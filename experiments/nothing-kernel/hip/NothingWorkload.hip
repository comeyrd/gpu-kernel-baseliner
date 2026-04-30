#include "../NothingWorkload.hpp"
#include "hip/hip_runtime.h"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>

#include <vector>

namespace Baseliner {

  __global__ void nothing_kernel(char * /*dummy*/) {
  }

  template <>
  void NothingWorkload<Hardware::HipBackend>::setup_device(typename backend::stream_t &stream) {
    std::vector<char> host(m_bytes_copied);
    if (m_async_memcpy) {
      CHECK_HIP(hipMallocAsync(&m_d_buffer, m_bytes_copied * sizeof(char), stream));
      CHECK_HIP(hipMemcpyAsync(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), hipMemcpyHostToDevice, stream));
    } else {
      CHECK_HIP(hipMalloc(&m_d_buffer, m_bytes_copied * sizeof(char)));
      CHECK_HIP(hipMemcpy(m_d_buffer, host.data(), m_bytes_copied * sizeof(char), hipMemcpyHostToDevice));
    }
  }

  template <>
  void NothingWorkload<Hardware::HipBackend>::reset_device(typename backend::stream_t &stream) {
    if (m_async_memcpy) {
      CHECK_HIP(hipMemsetAsync(m_d_buffer, 0, m_bytes_copied * sizeof(char), stream));
    } else {
      CHECK_HIP(hipMemset(m_d_buffer, 0, m_bytes_copied * sizeof(char)));
    }
  }

  template <>
  auto NothingWorkload<Hardware::HipBackend>::run(typename backend::stream_t &stream) -> std::monostate {
    nothing_kernel<<<m_blocks, m_threads, 0, stream>>>(m_d_buffer);
    return {};
  }

  template <>
  void NothingWorkload<Hardware::HipBackend>::fetch_results(typename backend::stream_t &stream) {
    CHECK_HIP(hipFree(m_d_buffer));
    m_d_buffer = nullptr;
  }
  namespace {
    using NothingWorkload = NothingWorkload<Hardware::HipBackend>;
    BASELINER_REGISTER_WORKLOAD(NothingWorkload);
  } // namespace

} // namespace Baseliner
