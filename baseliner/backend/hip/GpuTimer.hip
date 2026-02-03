#include <baseliner/backend/hip/HipBackend.hpp>
namespace Baseliner {
  namespace Backend {

    void HipBackend::GpuTimer::start() {
      CHECK_HIP(hipEventRecord(m_start_event, *m_stream));
    }
    void HipBackend::GpuTimer::stop() {
      CHECK_HIP(hipEventRecord(m_stop_event, *m_stream));
    }
    std::chrono::duration<float, std::milli> HipBackend::GpuTimer::time_elapsed() {
      float result;
      CHECK_HIP(hipEventSynchronize(m_stop_event));
      CHECK_HIP(hipEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    }
  } // namespace Backend

} // namespace Baseliner
