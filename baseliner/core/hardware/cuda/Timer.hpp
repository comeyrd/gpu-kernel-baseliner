#pragma once
#include <baseliner/core/Timer.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
namespace Baseliner::Hardware {
  template <>
  class GpuTimer<CudaBackend> : public ITimer<CudaBackend> {
  public:
    ~GpuTimer() {
      for (size_t index = 0; index < m_start_event.capacity(); index++) {
        CHECK_CUDA(cudaEventDestroy(m_start_event[index]));
        CHECK_CUDA(cudaEventDestroy(m_stop_event[index]));
      }
    };
    GpuTimer(std::shared_ptr<CudaBackend::stream_t> stream, size_t batch_size = 25)
        : ITimer<CudaBackend>(stream, batch_size) {
      m_start_event.resize(batch_size);
      m_stop_event.resize(batch_size);
      for (size_t index = 0; index < m_start_event.capacity(); index++) {
        CHECK_CUDA(cudaEventCreate(&m_start_event[index]));
        CHECK_CUDA(cudaEventCreate(&m_stop_event[index]));
      }
      m_start_p = 0;
      m_stop_p = 0;
    }
    GpuTimer(const GpuTimer &) = delete;
    auto operator=(const GpuTimer &) -> GpuTimer & = delete;
    GpuTimer(GpuTimer &&) = delete;
    auto operator=(GpuTimer &&) -> GpuTimer & = delete;
    void start() override;
    void stop() override;
    auto time_elapsed() -> std::vector<float_milliseconds> override;

  private:
    size_t m_start_p;
    size_t m_stop_p;
    std::vector<cudaEvent_t> m_start_event;
    std::vector<cudaEvent_t> m_stop_event;
  };
} // namespace Baseliner::Hardware