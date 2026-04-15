#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
#include <baseliner/core/hardware/cuda/Timer.hpp>
namespace Baseliner::Hardware {

  void GpuTimer<CudaBackend>::start() {
    if (m_start_p != m_stop_p) {
      throw Errors::timer_start_before_stop();
    }
    if (m_start_p >= m_batchsize) {
      throw Errors::timer_exhausted_batch_size_events();
    }
    CHECK_CUDA(cudaEventRecord(m_start_event[m_start_p], *this->m_stream));
    m_start_p++;
  };
  void GpuTimer<CudaBackend>::stop() {
    if ((m_start_p - 1) != m_stop_p) {
      throw Errors::timer_stop_before_start();
    }
    if (m_stop_p >= m_batchsize) {
      throw Errors::timer_exhausted_batch_size_events();
    }
    CHECK_CUDA(cudaEventRecord(m_stop_event[m_stop_p], *this->m_stream));
    m_stop_p++;
  };
  auto GpuTimer<CudaBackend>::time_elapsed() -> std::vector<float_milliseconds> {
    if (m_start_p != m_stop_p) {
      throw Errors::timer_elapsed_before_stop();
    }
    std::vector<float_milliseconds> results;
    results.resize(m_start_p);
    CHECK_CUDA(cudaEventSynchronize(m_stop_event[m_start_p - 1])); // Get the last element(m_start_p -1)
    for (size_t index = 0; index < m_start_p; index++) {
      float temp_f{};
      CHECK_CUDA(cudaEventElapsedTime(&temp_f, m_start_event[index], m_stop_event[index]));
      results[index] = float_milliseconds(temp_f);
    }
    m_start_p = 0;
    m_stop_p = 0;
    return results;
  };

} // namespace Baseliner::Hardware