#include "HipBackend.hpp"
namespace Baseliner {
  namespace Backend {

    void HipBackend::GpuTimer::start(hipStream_t stream) {
      CHECK_HIP(hipEventRecord(start_event, stream));
    }
    void HipBackend::GpuTimer::stop(hipStream_t stream) {
      CHECK_HIP(hipEventRecord(stop_event, stream));
    }
    float HipBackend::GpuTimer::time_elapsed() {
      float result;
      CHECK_HIP(hipEventSynchronize(stop_event));
      CHECK_HIP(hipEventElapsedTime(&result, start_event, stop_event));
      return result;
    }

    HipBackend::GpuTimer::GpuTimer() {
      CHECK_HIP(hipEventCreate(&start_event));
      CHECK_HIP(hipEventCreate(&stop_event));
    }
    HipBackend::GpuTimer::~GpuTimer() {
      CHECK_HIP(hipEventDestroy(start_event));
      CHECK_HIP(hipEventDestroy(stop_event));
    }
  } // namespace Backend

} // namespace Baseliner
