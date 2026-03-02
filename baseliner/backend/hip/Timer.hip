#include <baseliner/backend/hip/HipBackend.hpp>
namespace Baseliner::Backend {

  GpuTimer<HipBackend>::GpuTimer() {
    int current_device = HipBackend::instance()->get_current_device();
    int max_device = HipBackend::get_device_count();
    m_start_event.resize(max_device);
    m_stop_event.resize(max_device);
    for (int device = 0; device < max_device; device++) {
      HipBackend::set_device(device);
      alloc(device);
    }
    HipBackend::set_device(current_device);
  }
  GpuTimer<HipBackend>::~GpuTimer() {
    int current_device = HipBackend::instance()->get_current_device();
    int max_device = HipBackend::get_device_count();
    for (int device = 0; device < max_device; device++) {
      HipBackend::set_device(device);
      free(device);
    }
    HipBackend::set_device(current_device);
  }

  void GpuTimer<HipBackend>::alloc(int device) {
    CHECK_HIP(hipEventCreate(&m_start_event[device]));
    CHECK_HIP(hipEventCreate(&m_stop_event[device]));
  }
  void GpuTimer<HipBackend>::free(int device) {
    CHECK_HIP(hipEventDestroy(m_start_event[device]));
    CHECK_HIP(hipEventDestroy(m_stop_event[device]));
  }
  void GpuTimer<HipBackend>::measure_start(std::shared_ptr<HipBackend::stream_t> stream) {
    int device = HipBackend::instance()->get_current_device();
    CHECK_HIP(hipEventRecord(m_start_event[device], *stream));
  };
  void GpuTimer<HipBackend>::measure_stop(std::shared_ptr<HipBackend::stream_t> stream) {
    int device = HipBackend::instance()->get_current_device();
    CHECK_HIP(hipEventRecord(m_stop_event[device], *stream));
  };
  auto GpuTimer<HipBackend>::time_elapsed() -> float_milliseconds {
    float result{};
    int device = HipBackend::instance()->get_current_device();
    CHECK_HIP(hipEventSynchronize(m_stop_event[device]));
    CHECK_HIP(hipEventElapsedTime(&result, m_start_event[device], m_stop_event[device]));
    return float_milliseconds(result);
  };

} // namespace Baseliner::Backend
