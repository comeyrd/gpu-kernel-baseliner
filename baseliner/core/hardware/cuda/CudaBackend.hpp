#ifndef BASELINER_CORE_HARDWARE_CUDA_CUDABACKEND_HPP
#define BASELINER_CORE_HARDWARE_CUDA_CUDABACKEND_HPP
#include "cuda_runtime.h"
#include <baseliner/core/Benchmark.hpp>
#include <baseliner/core/Kernel.hpp>
#include <baseliner/core/Timer.hpp>
#include <baseliner/core/hardware/Backend.hpp>

void check_cuda_error(cudaError_t error_code, const char *file, int line);                // NOLINT
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Hardware {
    using CudaBackend = Backend<cudaStream_t, std::monostate>;
  } // namespace Hardware
  using ICudaWorkload = IWorkload<Hardware::CudaBackend>;
  using CudaBenchmark = Benchmark<Hardware::CudaBackend>;

  template <typename Input, typename Output>
  using ICudaKernel = IKernel<Hardware::CudaBackend, Input, Output>;
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