#include <baseliner/backend/hip/HipBackend.hpp>
namespace Baseliner {
  namespace Device {

    template <>
    L2Flusher<HipBackend>::L2Flusher() {
      int dev_id{};
      CHECK_HIP(hipGetDevice(&dev_id));
      CHECK_HIP(hipDeviceGetAttribute(&m_buffer_size, hipDeviceAttributeL2CacheSize, dev_id));
      if (m_buffer_size > 0) {
        CHECK_HIP(hipMalloc(&m_l2_buffer, static_cast<std::size_t>(m_buffer_size)));
      }
    }
    template <>
    L2Flusher<HipBackend>::~L2Flusher() {
      if (m_l2_buffer != nullptr) {
        CHECK_HIP(hipFree(m_l2_buffer));
      }
    }

    template <>
    void L2Flusher<HipBackend>::flush(std::shared_ptr<typename HipBackend::stream_t> stream) {
      if (m_l2_buffer != nullptr) {
        CHECK_HIP(hipMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_buffer_size), *stream));
      }
    }
  } // namespace Device

} // namespace Baseliner
