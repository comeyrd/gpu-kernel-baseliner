#include <baseliner/hardware/hip/HipBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
void check_hip_error(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    std::string msg = std::string("HIP Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}
void check_hip_error_no_except(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    std::string msg = std::string("HIP Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
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
    void HipBackend::synchronize(std::shared_ptr<hipStream_t> stream) {
      CHECK_HIP(hipStreamSynchronize(*stream));
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
      auto *stream = new HipBackend::stream_t;
      CHECK_HIP(hipStreamCreate(stream));

      return std::shared_ptr<hipStream_t>(stream, [](hipStream_t *s) {
        if (s) {
          CHECK_HIP_NO_EXCEPT(hipStreamDestroy(*s));
          delete s;
        }
      });
    }
    BASELINER_REGISTER_BACKEND("hip", HipBackend);
  } // namespace Hardware

} // namespace Baseliner
