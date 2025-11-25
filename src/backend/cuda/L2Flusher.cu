#include "CudaBackend.hpp"
namespace Baseliner {
  namespace Backend {

    CudaBackend::L2Flusher::L2Flusher() {
      int dev_id{};
      CHECK_CUDA(cudaGetDevice(&dev_id));
      CHECK_CUDA(cudaDeviceGetAttribute(&buffer_size, cudaDevAttrL2CacheSize, dev_id));
      if (buffer_size > 0) {
        void *buffer = l2_buffer;
        CHECK_CUDA(cudaMalloc(&buffer, static_cast<std::size_t>(buffer_size)));
        l2_buffer = reinterpret_cast<int *>(buffer);
      }
    }
    CudaBackend::L2Flusher::~L2Flusher() {
    }

    void CudaBackend::L2Flusher::flush(cudaStream_t stream) {
    }
  } // namespace Backend

} // namespace Baseliner
