#include "CudaBackend.hpp"
void check_cuda_error(cudaError_t error_code, const char *file, int line) {
  if (error_code != cudaSuccess) {
    std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file +
                      std::string(" line ") + std::to_string(line);
    throw std::runtime_error(msg);
  }
}
namespace Baseliner {
  namespace Backend {
    void CudaBackend::set_device(int device) {
      CHECK_CUDA(cudaSetDevice(device));
    }
    void CudaBackend::reset_device() {
      CHECK_CUDA(cudaDeviceReset());
    }

    void CudaBackend::synchronize(cudaStream_t stream) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

  } // namespace Backend

} // namespace Baseliner
