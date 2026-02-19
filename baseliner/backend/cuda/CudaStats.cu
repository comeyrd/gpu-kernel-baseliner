#include <baseliner/backend/BackendStats.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <baseliner/stats/Stats.hpp>

namespace Baseliner::Stats {

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
  template <>
  std::string ClockFrequency<Device::CudaBackend>::name() const {
    return "CudaClockFrequency";
  }
  template <>
  void ClockFrequency<Device::CudaBackend>::calculate(ClockFrequency<Device::CudaBackend>::type &value_to_update) {
    NvmlManager::ensure_init();
    nvmlDevice_t device = NvmlManager::get_current_device();
    unsigned int clockMHz = 0;
    CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clockMHz));
    value_to_update = static_cast<float>(clockMHz) / 1000;
  }
  template <>
  std::string ClockFrequencyVector<Device::CudaBackend>::name() const {
    return "CudaClockFrequencyVector";
  }
#endif
} // namespace Baseliner::Stats
