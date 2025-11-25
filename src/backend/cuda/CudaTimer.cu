#include "CudaBackend.hpp"
namespace Baseliner {
namespace Backend {

void CudaBackend::GpuTimer::start(cudaStream_t stream){
  CHECK_CUDA(cudaEventRecord(start_event,stream));
}
void CudaBackend::GpuTimer::stop(cudaStream_t stream){
  CHECK_CUDA(cudaEventRecord(stop_event,stream));
}
float CudaBackend::GpuTimer::time_elapsed(){
  float result;
  CHECK_CUDA(cudaEventSynchronize(stop_event));
  CHECK_CUDA(cudaEventElapsedTime(&result, start_event, stop_event));
  return result;
}

CudaBackend::GpuTimer::GpuTimer(){
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&stop_event));
}
CudaBackend::GpuTimer::~GpuTimer(){
  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(stop_event));
}
} // namespace Backend

} // namespace Baseliner
