#include <baseliner/backend/cuda/CudaBackend.hpp>
namespace Baseliner::Backend {

  GpuTimer<CudaBackend>::GpuTimer() {
    int current_device = CudaBackend::instance()->get_current_device();
    int max_device = CudaBackend::get_device_count();
    m_start_event.resize(max_device);
    m_stop_event.resize(max_device);
    for (int device = 0; device < max_device; device++) {
      CudaBackend::set_device(device);
      alloc(device);
    }
    CudaBackend::set_device(current_device);
  }
  GpuTimer<CudaBackend>::~GpuTimer() {
    int current_device = CudaBackend::instance()->get_current_device();
    int max_device = CudaBackend::get_device_count();
    for (int device = 0; device < max_device; device++) {
      CudaBackend::set_device(device);
      free(device);
    }
    CudaBackend::set_device(current_device);
  }

  void GpuTimer<CudaBackend>::alloc(int device) {
    CHECK_CUDA(cudaEventCreate(&m_start_event[device]));
    CHECK_CUDA(cudaEventCreate(&m_stop_event[device]));
  }
  void GpuTimer<CudaBackend>::free(int device) {
    CHECK_CUDA(cudaEventDestroy(m_start_event[device]));
    CHECK_CUDA(cudaEventDestroy(m_stop_event[device]));
  }
  void GpuTimer<CudaBackend>::measure_start(std::shared_ptr<CudaBackend::stream_t> stream) {
    int device = CudaBackend::instance()->get_current_device();
    CHECK_CUDA(cudaEventRecord(m_start_event[device], *stream));
  };
  void GpuTimer<CudaBackend>::measure_stop(std::shared_ptr<CudaBackend::stream_t> stream) {
    int device = CudaBackend::instance()->get_current_device();
    CHECK_CUDA(cudaEventRecord(m_stop_event[device], *stream));
  };
  auto GpuTimer<CudaBackend>::time_elapsed() -> float_milliseconds {
    float result{};
    int device = CudaBackend::instance()->get_current_device();
    CHECK_CUDA(cudaEventSynchronize(m_stop_event[device]));
    CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event[device], m_stop_event[device]));
    return float_milliseconds(result);
  };

} // namespace Baseliner::Backend