#include <baseliner/hardware/BackendStats.hpp>
#include <baseliner/hardware/cuda/CudaBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/stats/Stats.hpp>
namespace Baseliner::Stats {

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
  template <>
  std::string ClockFrequency<Hardware::CudaBackend>::name() const {
    return "CudaClockFrequency";
  }
  template <>
  void ClockFrequency<Hardware::CudaBackend>::calculate(ClockFrequency<Hardware::CudaBackend>::type &value_to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t Backend = NvmlManager::get_current_device();
    unsigned int clockMHz = 0;
    CHECK_NVML(nvmlDeviceGetClockInfo(Backend, NVML_CLOCK_SM, &clockMHz));
    value_to_update = static_cast<float>(clockMHz) / 1000;
  }
  template <>
  std::string ClockFrequencyVector<Hardware::CudaBackend>::name() const {
    return "CudaClockFrequencyVector";
  }
  namespace {
    using ClockFrequency = ClockFrequency<Hardware::CudaBackend>;
    using ClockFrequencyVector = ClockFrequencyVector<Hardware::CudaBackend>;

    BASELINER_REGISTER_BACKEND_STATS(ClockFrequency);
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequencyVector);
  } // namespace
#endif
} // namespace Baseliner::Stats
