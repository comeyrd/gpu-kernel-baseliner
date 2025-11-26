#include "HipBackend.hpp"
namespace Baseliner {
  namespace Backend {

    void HipBackend::GpuTimer::start(hipStream_t stream) {
      CHECK_HIP(hipEventRecord(m_start_event, stream));
    }
    void HipBackend::GpuTimer::stop(hipStream_t stream) {
      CHECK_HIP(hipEventRecord(m_stop_event, stream));
    }
    float_milliseconds HipBackend::GpuTimer::time_elapsed() {
      float result;
      CHECK_HIP(hipEventSynchronize(m_stop_event));
      CHECK_HIP(hipEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    }

    HipBackend::GpuTimer::GpuTimer() {
      CHECK_HIP(hipEventCreate(&m_start_event));
      CHECK_HIP(hipEventCreate(&m_stop_event));
    }
    HipBackend::GpuTimer::~GpuTimer() {
      CHECK_HIP(hipEventDestroy(m_start_event));
      CHECK_HIP(hipEventDestroy(m_stop_event));
    }
  } // namespace Backend

} // namespace Baseliner
