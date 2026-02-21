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
  namespace Backend {
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
  } // namespace Backend
  using ICudaCase = ICase<Backend::CudaBackend>;
  using CudaBenchmark = Benchmark<Backend::CudaBackend>;

  template <typename Input, typename Output>
  using ICudaKernel = IKernel<Backend::CudaBackend, Input, Output>;
} // namespace Baseliner

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
void check_nvml_error(nvmlReturn_t error_code, const char *file, int line);               // NOLINT
void check_nvml_error_no_except(nvmlReturn_t error_code, const char *file, int line);     // NOLINT
#define CHECK_NVML(error) check_nvml_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_NVML_NO_EXCEPT(error) check_nvml_error_no_except(error, __FILE__, __LINE__) // NOLINT
class NvmlManager {
public:
  // This is called automatically the first time Instance() is accessed
  NvmlManager() {
    nvmlInit();
  }

  // This is called when the program exits
  ~NvmlManager() {
    nvmlShutdown();
  }
  static auto get_current_device() -> nvmlDevice_t {
    ensure_init();

    int cudaIdx = 0;
    CHECK_CUDA(cudaGetDevice(&cudaIdx));

    char pciBusId[64];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pciBusId, 64, cudaIdx));

    // 3. Ask NVML for the handle matching that specific PCI Bus ID
    nvmlDevice_t Backend;
    CHECK_NVML(nvmlDeviceGetHandleByPciBusId(pciBusId, &Backend));
    return Backend;
  }

  static void ensure_init() {
    static NvmlManager instance;
  }
};
#endif
#endif // CUDA_BACKEND_HPP