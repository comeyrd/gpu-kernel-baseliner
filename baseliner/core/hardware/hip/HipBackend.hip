#include <baseliner/core/Error.hpp>

#include <baseliner/core/hardware/hip/HipBackend.hpp>
void check_hip_error(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    throw Baseliner::Errors::hardware_error("HIP", hipGetErrorString(error_code), file, line);
  }
}
void check_hip_error_no_except(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    auto msg = Baseliner::Errors::hardware_error_noexcept("HIP", hipGetErrorString(error_code), file, line);
    std::cerr << msg << std::endl;
  }
}
namespace Baseliner {
  namespace Hardware {
    template <>
    void HipBackend::set_device(int device) {
      CHECK_HIP(hipSetDevice(device));
    }
    template <>
    void HipBackend::reset_device() {
      CHECK_HIP(hipDeviceReset());
    }
    template <>
    void HipBackend::synchronize(hipStream_t &stream) {
      CHECK_HIP(hipStreamSynchronize(stream));
    }
    template <>
    auto HipBackend::get_device_count() -> int {
      int device_c{};
      CHECK_HIP(hipGetDeviceCount(&device_c));
      return device_c;
    }
    template <>
    auto HipBackend::get_device_info() -> HardwareInfo {
      hipDeviceProp_t prop;
      CHECK_HIP(hipGetDeviceProperties(&prop, this->get_current_device()));
      return HardwareInfo{prop.name};
    }
    template <>
    void HipBackend::get_last_error() {
      CHECK_HIP(hipGetLastError());
    }
    template <>
    auto HipBackend::inner_create_stream() -> std::shared_ptr<HipBackend::stream_t> {
      auto stream = new HipBackend::stream_t;
      CHECK_HIP(hipStreamCreate(stream));

      return std::make_shared<hipStream_t>(stream, [](hipStream_t *s) {
        if (s) {
          CHECK_HIP_NO_EXCEPT(hipStreamDestroy(*s));
          delete s;
        }
      });
    }

#ifdef BASELINER_HAS_AMDSMI
    template <>
    void HipBackend::lock_clocks(int min_clock_val, int max_clock_val) {
      AmdSmiManager *manager = AmdSmiManager::ensure_init();
      amdsmi_processor_handle handle = manager->get_current_device();
      amdsmi_frequencies_t freq_info;
      CHECK_AMDSMI(amdsmi_get_clk_freq(handle, AMDSMI_CLK_TYPE_SYS, &freq_info));
      amdsmi_set_gpu_perf_level(handle, AMDSMI_DEV_PERF_LEVEL_DETERMINISM);
    };
    template <>
    void HipBackend::unlock_clock() {
      AmdSmiManager *manager = AmdSmiManager::ensure_init();
      amdsmi_processor_handle handle = manager->get_current_device();
      amdsmi_set_gpu_perf_level(handle, AMDSMI_DEV_PERF_LEVEL_AUTO);
    };
#else
    template <>
    void HipBackend::unlock_clock() {
      throw Errors::not_found_in_backend("AMDSMI", "Locking clocks", "hip");
    };
    template <>
    void HipBackend::lock_clocks(int min_clock_val, int max_clock_val) {
      // throw Errors::not_found_in_backend("NVML", "unlocking_clocks", "hip");
    };
#endif
  } // namespace Hardware

} // namespace Baseliner
