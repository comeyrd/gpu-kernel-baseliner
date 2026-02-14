#include <baseliner/backend/cuda/CudaBackend.hpp>
namespace Baseliner::Device {

  GpuTimer<CudaBackend>::GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&m_start_event));
    CHECK_CUDA(cudaEventCreate(&m_stop_event));
  }
  GpuTimer<CudaBackend>::~GpuTimer() {
    CHECK_CUDA(cudaEventDestroy(m_start_event));
    CHECK_CUDA(cudaEventDestroy(m_stop_event));
  }
  void GpuTimer<CudaBackend>::measure_start(std::shared_ptr<CudaBackend::stream_t> stream) {
    CHECK_CUDA(cudaEventRecord(m_start_event, *stream));
  };
  void GpuTimer<CudaBackend>::measure_stop(std::shared_ptr<CudaBackend::stream_t> stream) {
    CHECK_CUDA(cudaEventRecord(m_stop_event, *stream));
  };
  auto GpuTimer<CudaBackend>::time_elapsed() -> float_milliseconds {
    float result{};
    CHECK_CUDA(cudaEventSynchronize(m_stop_event));
    CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
    return float_milliseconds(result);
  };

} // namespace Baseliner::Device