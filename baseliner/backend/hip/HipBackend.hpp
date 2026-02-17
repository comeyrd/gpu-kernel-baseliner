#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP
#include "baseliner/Kernel.hpp"
#include "hip/hip_runtime.h"
#include <baseliner/Benchmark.hpp>
#include <baseliner/backend/Backend.hpp>

void check_hip_error(hipError_t error_code, const char *file, int line);                // NOLINT
void check_hip_error_no_except(hipError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Device {
    using HipBackend = Backend<hipStream_t>;
    template <>
    class GpuTimer<HipBackend> {
    public:
      ~GpuTimer();
      GpuTimer();
      GpuTimer(const GpuTimer &) = delete;
      auto operator=(const GpuTimer &) -> GpuTimer & = delete;
      GpuTimer(GpuTimer &&) = delete;
      auto operator=(GpuTimer &&) -> GpuTimer & = delete;

      void measure_start(std::shared_ptr<typename HipBackend::stream_t> stream);
      void measure_stop(std::shared_ptr<typename HipBackend::stream_t> stream);
      auto time_elapsed() -> float_milliseconds;

    protected:
    private:
      hipEvent_t m_start_event{};
      hipEvent_t m_stop_event{};
    };
  } // namespace Device
  using IHipCase = ICase<Device::HipBackend>;
  using HipBenchmark = Benchmark<Device::HipBackend>;

  template <typename Input, typename Output>
  using IHipKernel = IKernel<Device::HipBackend, Input, Output>;
} // namespace Baseliner

#endif // HIP_BACKEND_HPP
