#include "HipBackend.hpp"

void check_hip_error(hipError_t error_code, const char *file, int line) {
  if (error_code != hipSuccess) {
    std::string msg = std::string("CUDA Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}

namespace Baseliner {
  namespace Backend {
    void HipBackend::set_device(int device) {
      CHECK_HIP(hipSetDevice(device));
    }
    void HipBackend::reset_device() {
      CHECK_HIP(hipDeviceReset());
    }

    void HipBackend::synchronize(hipStream_t stream) {
      CHECK_HIP(hipStreamSynchronize(stream));
    }

  } // namespace Backend

} // namespace Baseliner