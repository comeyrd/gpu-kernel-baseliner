#include "CudaBackend.hpp"
namespace Baseliner {
namespace Backend {

void CudaBackend::GpuTimer::start(cudaStream_t stream){
  CHECK_CUDA(cudaEventRecord(m_start_event,stream));
}
void CudaBackend::GpuTimer::stop(cudaStream_t stream){
  CHECK_CUDA(cudaEventRecord(m_stop_event,stream));
}
std::chrono::duration<float, std::milli> CudaBackend::GpuTimer::time_elapsed(){
  float result;
  CHECK_CUDA(cudaEventSynchronize(m_stop_event));
  CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
  return float_milliseconds(result);
}

CudaBackend::GpuTimer::GpuTimer(){
  CHECK_CUDA(cudaEventCreate(&m_start_event));
  CHECK_CUDA(cudaEventCreate(&m_stop_event));
}
CudaBackend::GpuTimer::~GpuTimer(){
  CHECK_CUDA(cudaEventDestroy(m_start_event));
  CHECK_CUDA(cudaEventDestroy(m_stop_event));
}
} // namespace Backend

} // namespace Baseliner
