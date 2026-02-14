#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include "baseliner/Kernel.hpp"
#include "cuda_runtime.h"
#include <baseliner/Benchmark.hpp>
#include <baseliner/backend/Backend.hpp>

void check_cuda_error(cudaError_t error_code, const char *file, int line);                // NOLINT
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Device {
    using CudaBackend = Backend<cudaStream_t>;
    template <>
    class GpuTimer<CudaBackend> {
    public:
      ~GpuTimer();
      GpuTimer();
      GpuTimer(const GpuTimer &) = delete;
      auto operator=(const GpuTimer &) -> GpuTimer & = delete;
      GpuTimer(GpuTimer &&) = delete;
      auto operator=(GpuTimer &&) -> GpuTimer & = delete;

      void measure_start(std::shared_ptr<typename CudaBackend::stream_t> stream);
      void measure_stop(std::shared_ptr<typename CudaBackend::stream_t> stream);
      auto time_elapsed() -> float_milliseconds;

    protected:
    private:
      cudaEvent_t m_start_event{};
      cudaEvent_t m_stop_event{};
    };
  } // namespace Device
  using ICudaCase = ICase<Device::CudaBackend>;
  using CudaBenchmark = Benchmark<Device::CudaBackend>;

  template <typename Input, typename Output>
  using ICudaKernel = IKernel<Device::CudaBackend, Input, Output>;
} // namespace Baseliner

#endif // CUDA_BACKEND_HPP