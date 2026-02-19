#include <baseliner/backend/hip/HipBackend.hpp>
#include<baseliner/backend/BackendStats.hpp>
namespace Baseliner::Stats {
#ifdef BASELINER_HAS_AMDSMI
constexpr size_t KHZ_TO_HZ = 1000;
constexpr size_t MHZ_TO_KHZ = 1000;
constexpr size_t GHZ_TO_MHZ = 1000;
#include "amd_smi/amdsmi.h"
  template <>
  std::string ClockFrequency<Device::HipBackend>::name() const {
    return "HipClockFrequency";
  }
  template <>
  void ClockFrequency<Device::HipBackend>::calculate(ClockFrequency<Device::HipBackend>::type &value_to_update) {
    AmdSmiManager* manager = AmdSmiManager::ensure_init();
    

    amdsmi_processor_handle handle = manager->get_current_device();
    amdsmi_frequencies_t info;
    CHECK_AMDSMI(amdsmi_get_clk_freq(handle, amdsmi_clk_type_t::AMDSMI_CLK_TYPE_SYS, &info));
    value_to_update = static_cast<float>(info.frequency[info.current])/(GHZ_TO_MHZ*MHZ_TO_KHZ*KHZ_TO_HZ);
  }
  template <>
  std::string ClockFrequencyVector<Device::HipBackend>::name() const {
    return "HipClockFrequencyVector";
  }
#endif
} // namespace Baseliner::Stats
