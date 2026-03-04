#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/stats/Stats.hpp>
namespace Baseliner::Stats {

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
  template <>
  void ClockFrequency<Hardware::CudaBackend>::calculate(ClockFrequency<Hardware::CudaBackend>::type &value_to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t Backend = NvmlManager::get_current_device();
    unsigned int clockMHz = 0;
    CHECK_NVML(nvmlDeviceGetClockInfo(Backend, NVML_CLOCK_SM, &clockMHz));
    value_to_update = static_cast<float>(clockMHz) / 1000;
  }
  template <>
  void DeviceTemperature<Hardware::CudaBackend>::calculate(
      typename DeviceTemperature<Hardware::CudaBackend>::type &value_to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t Backend = NvmlManager::get_current_device();
    // nvmlTemperature_t temperature_thingy;
    // temperature_thingy.sensorType = nvmlTemperatureSensors_t::NVML_TEMPERATURE_GPU;
    //  CHECK_NVML(nvmlDeviceGetTemperatureV(Backend, &temperature_thingy));
    unsigned int temp = 0;
    CHECK_NVML(nvmlDeviceGetTemperature(Backend, NVML_TEMPERATURE_GPU, &temp));
    value_to_update = static_cast<int>(temp);
  }
  template <>
  void DevicePowerUtilization<Hardware::CudaBackend>::calculate(
      typename DevicePowerUtilization<Hardware::CudaBackend>::type &to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t Backend = NvmlManager::get_current_device();
    unsigned int mw_act_power;
    CHECK_NVML(nvmlDeviceGetPowerUsage(Backend, &mw_act_power));
    unsigned int mw_limit;

    CHECK_NVML(nvmlDeviceGetEnforcedPowerLimit(Backend, &mw_limit));
    to_update = (static_cast<float>(mw_act_power) / mw_limit) * 100.0F;
  }
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
