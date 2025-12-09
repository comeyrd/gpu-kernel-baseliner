#include "CudaBackend.hpp"
namespace Baseliner {
  namespace Backend {

    void CudaBackend::EventTimer::start() {
      CHECK_CUDA(cudaEventRecord(m_start_event, *m_stream));
    }
    void CudaBackend::EventTimer::stop() {
      CHECK_CUDA(cudaEventRecord(m_stop_event, *m_stream));
    }
    std::chrono::duration<float, std::milli> CudaBackend::EventTimer::time_elapsed() {
      float result;
      CHECK_CUDA(cudaEventSynchronize(m_stop_event));
      CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    }
  } // namespace Backend

} // namespace Baseliner
