#include <baseliner/backend/BackendStats.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
#include <baseliner/stats/Stats.hpp>
namespace Baseliner::Stats {

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
  template <>
  std::string ClockFrequency<Backend::CudaBackend>::name() const {
    return "CudaClockFrequency";
  }
  template <>
  void ClockFrequency<Backend::CudaBackend>::calculate(ClockFrequency<Backend::CudaBackend>::type &value_to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t Backend = NvmlManager::get_current_device();
    unsigned int clockMHz = 0;
    CHECK_NVML(nvmlDeviceGetClockInfo(Backend, NVML_CLOCK_SM, &clockMHz));
    value_to_update = static_cast<float>(clockMHz) / 1000;
  }
  template <>
  std::string ClockFrequencyVector<Backend::CudaBackend>::name() const {
    return "CudaClockFrequencyVector";
  }
  namespace {
    using ClockFrequency = ClockFrequency<Backend::CudaBackend>;
    using ClockFrequencyVector = ClockFrequencyVector<Backend::CudaBackend>;

    BASELINER_REGISTER_BACKEND_STATS(ClockFrequency);
    BASELINER_REGISTER_BACKEND_STATS(ClockFrequencyVector);
  } // namespace
#endif
} // namespace Baseliner::Stats
