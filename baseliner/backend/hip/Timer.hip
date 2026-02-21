#include <baseliner/backend/hip/HipBackend.hpp>
namespace Baseliner::Backend {

  GpuTimer<HipBackend>::GpuTimer() {
    CHECK_HIP(hipEventCreate(&m_start_event));
    CHECK_HIP(hipEventCreate(&m_stop_event));
  }
  GpuTimer<HipBackend>::~GpuTimer() {
    CHECK_HIP(hipEventDestroy(m_start_event));
    CHECK_HIP(hipEventDestroy(m_stop_event));
  }
  void GpuTimer<HipBackend>::measure_start(std::shared_ptr<HipBackend::stream_t> stream) {
    CHECK_HIP(hipEventRecord(m_start_event, *stream));
  };
  void GpuTimer<HipBackend>::measure_stop(std::shared_ptr<HipBackend::stream_t> stream) {
    CHECK_HIP(hipEventRecord(m_stop_event, *stream));
  };
  auto GpuTimer<HipBackend>::time_elapsed() -> float_milliseconds {
    float result{};
    CHECK_HIP(hipEventSynchronize(m_stop_event));
    CHECK_HIP(hipEventElapsedTime(&result, m_start_event, m_stop_event));
    return float_milliseconds(result);
  };

} // namespace Baseliner::Backend
