#include <baseliner/hardware/hip/HipBackend.hpp>
namespace Baseliner {
  namespace Hardware {

    template <>
    void L2Flusher<HipBackend>::alloc(int device) {
      CHECK_HIP(hipDeviceGetAttribute(&m_buffer_size_v[device], hipDeviceAttributeL2CacheSize, device));
      if (m_buffer_size_v[device] > 0) {
        CHECK_HIP(hipMalloc(&m_l2_buffer_v[device], static_cast<std::size_t>(m_buffer_size_v[device])));
      }
    }
    template <>
    void L2Flusher<HipBackend>::free(int device) {
      CHECK_HIP(hipFree(m_l2_buffer_v[device]));
    }

    template <>
    void L2Flusher<HipBackend>::flush(std::shared_ptr<typename HipBackend::stream_t> stream) {
      int current_device = HipBackend::instance()->get_current_device();
      CHECK_HIP(hipMemsetAsync(m_l2_buffer_v[current_device], 0,
                               static_cast<std::size_t>(m_buffer_size_v[current_device]), *stream));
    }
  } // namespace Hardware

} // namespace Baseliner
