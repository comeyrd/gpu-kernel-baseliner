#include <baseliner/backend/hip/HipBackend.hpp>
namespace Baseliner {
  namespace Backend {

    HipBackend::L2Flusher::L2Flusher() {
      int dev_id{};
      CHECK_HIP(hipGetDevice(&dev_id));
      CHECK_HIP(hipDeviceGetAttribute(&m_buffer_size, hipDeviceAttributeL2CacheSize, dev_id));
      if (m_buffer_size > 0) {
        void *buffer = m_l2_buffer;
        CHECK_HIP(hipMalloc(&buffer, static_cast<std::size_t>(m_buffer_size)));
        m_l2_buffer = reinterpret_cast<int *>(buffer);
      }
    }
    HipBackend::L2Flusher::~L2Flusher() {
      if (m_l2_buffer) {
        CHECK_HIP(hipFree(m_l2_buffer));
      }
    }

    void HipBackend::L2Flusher::flush(std::shared_ptr<hipStream_t> stream) {
      CHECK_HIP(hipMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_buffer_size), *stream));
    }
  } // namespace Backend

} // namespace Baseliner
