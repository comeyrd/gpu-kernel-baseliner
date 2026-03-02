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
  std::string ClockFrequency<Hardware::HipBackend>::name() const {
    return "HipClockFrequency";
  }
  template <>
  void ClockFrequency<Hardware::HipBackend>::calculate(ClockFrequency<Hardware::HipBackend>::type &value_to_update) {
    AmdSmiManager *manager = AmdSmiManager::ensure_init();

    amdsmi_processor_handle handle = manager->get_current_device();
    amdsmi_frequencies_t info;
    CHECK_AMDSMI(amdsmi_get_clk_freq(handle, amdsmi_clk_type_t::AMDSMI_CLK_TYPE_SYS, &info));
    value_to_update = static_cast<float>(info.frequency[info.current]) / (GHZ_TO_MHZ * MHZ_TO_KHZ * KHZ_TO_HZ);
  }
  template <>
  std::string ClockFrequencyVector<Hardware::HipBackend>::name() const {
    return "HipClockFrequencyVector";
  }
  namespace {
    using ClockFrequency = ClockFrequency<Hardware::HipBackend>;
    using ClockFrequencyVector = ClockFrequencyVector<Hardware::HipBackend>;

    BASELINER_REGISTER_BACKEND_STATS(ClockFrequency);
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequencyVector);
  } // namespace
#endif
} // namespace Baseliner::Stats
