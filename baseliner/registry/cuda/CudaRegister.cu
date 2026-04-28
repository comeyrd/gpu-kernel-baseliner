#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/BackendStats.hpp>
#include <baseliner/core/hardware/cuda/CudaBackend.hpp>
namespace Baseliner::Hardware {
  BASELINER_REGISTER_BACKEND("cuda", CudaBackend);
}
#ifdef BASELINER_HAS_NVML
namespace Baseliner::Stats {
  namespace {
    using ClockFrequency = ClockFrequency<Hardware::CudaBackend>;
    using ClockFrequencyVector = ClockFrequencyVector<Hardware::CudaBackend>;
    using DeviceTemperature = DeviceTemperature<Hardware::CudaBackend>;
    using DeviceTemperatureVector = DeviceTemperatureVector<Hardware::CudaBackend>;
    using DevicePowerUtilization = DevicePowerUtilization<Hardware::CudaBackend>;
    using DevicePowerUtilizationVector = DevicePowerUtilizationVector<Hardware::CudaBackend>;
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequency);
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequencyVector);
    BASELINER_REGISTER_BACKEND_STATS(DeviceTemperature);
    BASELINER_REGISTER_BACKEND_STATS(DeviceTemperatureVector);
    BASELINER_REGISTER_BACKEND_STATS(DevicePowerUtilization);
    BASELINER_REGISTER_BACKEND_STATS(DevicePowerUtilizationVector);
  } // namespace
#endif
} // namespace Baseliner::Stats