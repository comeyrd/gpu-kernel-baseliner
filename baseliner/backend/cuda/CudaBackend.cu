#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>
void check_cuda_error(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    std::cerr << msg << std::endl;
  }
}
namespace Baseliner {
  namespace Backend {
    template <>
    void CudaBackend::set_device(int device) {
      CHECK_CUDA(cudaSetDevice(device));
    }
    template <>
    void CudaBackend::reset_device() {
      CHECK_CUDA(cudaDeviceReset());
    }
    template <>
    void CudaBackend::synchronize(std::shared_ptr<cudaStream_t> stream) {
      CHECK_CUDA(cudaStreamSynchronize(*stream));
    }
    template <>
    auto CudaBackend::get_device_count() -> int {
      int device_c{};
      CHECK_CUDA(cudaGetDeviceCount(&device_c));
      return device_c;
    }
    template <>
    void CudaBackend::get_last_error() {
      CHECK_CUDA(cudaGetLastError());
    }
    template <>
    auto CudaBackend::inner_create_stream() -> std::shared_ptr<CudaBackend::stream_t> {
      auto *stream = new CudaBackend::stream_t;
      CHECK_CUDA(cudaStreamCreate(stream));

      return std::shared_ptr<cudaStream_t>(stream, [](cudaStream_t *s) {
        if (s) {
          CHECK_CUDA_NO_EXCEPT(cudaStreamDestroy(*s));
          delete s;
        }
      });
    }
    BASELINER_REGISTER_BACKEND("cuda", CudaBackend);
  } // namespace Backend

} // namespace Baseliner
