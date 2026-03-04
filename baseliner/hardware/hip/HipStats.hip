#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/hardware/hip/HipBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
namespace Baseliner::Stats {
#ifdef BASELINER_HAS_AMDSMI
  constexpr size_t KHZ_TO_HZ = 1000;
  constexpr size_t MHZ_TO_KHZ = 1000;
  constexpr size_t GHZ_TO_MHZ = 1000;
#include "amd_smi/amdsmi.h"
  template <>
  void ClockFrequency<Hardware::HipBackend>::calculate(ClockFrequency<Hardware::HipBackend>::type &value_to_update) {
    AmdSmiManager *manager = AmdSmiManager::ensure_init();

    amdsmi_processor_handle handle = manager->get_current_device();
    amdsmi_frequencies_t info;
    CHECK_AMDSMI(amdsmi_get_clk_freq(handle, amdsmi_clk_type_t::AMDSMI_CLK_TYPE_SYS, &info));
    value_to_update = static_cast<float>(info.frequency[info.current]) / (GHZ_TO_MHZ * MHZ_TO_KHZ * KHZ_TO_HZ);
  }

  template <>
  void DeviceTemperature<Hardware::HipBackend>::calculate(
      typename DeviceTemperature<Hardware::HipBackend>::type &value_to_update) {
    AmdSmiManager *manager = AmdSmiManager::ensure_init();
    amdsmi_processor_handle handle = manager->get_current_device();
    int64_t temp = 0;

    CHECK_AMDSMI(amdsmi_get_temp_metric(handle, amdsmi_temperature_type_t::AMDSMI_TEMPERATURE_TYPE_EDGE,
                                        amdsmi_temperature_metric_t::AMDSMI_TEMP_CURRENT, &temp));
    value_to_update = static_cast<int>(temp);
  }
  template <>
  void DevicePowerUtilization<Hardware::HipBackend>::calculate(
      typename DevicePowerUtilization<Hardware::HipBackend>::type &to_update) {
    AmdSmiManager *manager = AmdSmiManager::ensure_init();
    amdsmi_processor_handle handle = manager->get_current_device();

    amdsmi_power_info_t info{};
    CHECK_AMDSMI(amdsmi_get_power_info(handle, &info));
    auto power_limit = static_cast<float>(info.power_limit);
    if (info.power_limit == 0) {
      to_update = 0.0F;
      return;
    }
    float s_power = static_cast<float>(info.socket_power);
    if (power_limit > 2000.0F) {
      power_limit = power_limit / 1e6F;
    }
    if (s_power > (power_limit * 1.5F)) {
      s_power = s_power / 1e6F;
    }

    to_update = (s_power / power_limit) * 100.0F;
  }
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
#endif
} // namespace Baseliner::Stats
