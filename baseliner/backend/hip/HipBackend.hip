#include <baseliner/backend/hip/HipBackend.hpp>
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
  namespace Backend {
    void HipBackend::set_device(int device) {
      CHECK_HIP(hipSetDevice(device));
    }
    void HipBackend::reset_device() {
      CHECK_HIP(hipDeviceReset());
    }

    void HipBackend::synchronize(std::shared_ptr<hipStream_t> stream) {
      CHECK_HIP(hipStreamSynchronize(*stream));
    }
    std::shared_ptr<hipStream_t> HipBackend::create_stream() {
      hipStream_t *stream = new hipStream_t;
      CHECK_HIP(hipStreamCreate(stream));

      return std::shared_ptr<hipStream_t>(stream, [](hipStream_t *s) {
        if (s) {
          CHECK_HIP_NO_EXCEPT(hipStreamDestroy(*s));
          delete s;
        }
      });
    }
  } // namespace Backend

} // namespace Baseliner
