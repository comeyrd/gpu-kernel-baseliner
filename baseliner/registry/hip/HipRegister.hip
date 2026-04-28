#include "hip/hip_runtime.h"
#include <baseliner/Register.hpp>
#include <baseliner/core/hardware/hip/HipBackend.hpp>
#ifdef BASELINER_HAS_AMDSMI
namespace Baseliner::Stats {
  namespace {
    using ClockFrequency = ClockFrequency<Hardware::HipBackend>;
    using ClockFrequencyVector = ClockFrequencyVector<Hardware::HipBackend>;
    using DeviceTemperature = DeviceTemperature<Hardware::HipBackend>;
    using DeviceTemperatureVector = DeviceTemperatureVector<Hardware::HipBackend>;
    using DevicePowerUtilization = DevicePowerUtilization<Hardware::HipBackend>;
    using DevicePowerUtilizationVector = DevicePowerUtilizationVector<Hardware::HipBackend>;

    BASELINER_REGISTER_BACKEND_STATS(ClockFrequency);
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequencyVector);
    BASELINER_REGISTER_BACKEND_STATS(DeviceTemperature);
    BASELINER_REGISTER_BACKEND_STATS(DeviceTemperatureVector);
    BASELINER_REGISTER_BACKEND_STATS(DevicePowerUtilization);
    BASELINER_REGISTER_BACKEND_STATS(DevicePowerUtilizationVector);
  } // namespace
} // namespace Baseliner::Stats
#endif
namespace Baseliner::Hardware {
  BASELINER_REGISTER_BACKEND("hip", HipBackend);

}