#include "CudaBackend.hpp"
namespace Baseliner {
  namespace Backend {

    CudaBackend::L2Flusher::L2Flusher() {
      int dev_id{};
      CHECK_CUDA(cudaGetDevice(&dev_id));
      CHECK_CUDA(cudaDeviceGetAttribute(&m_buffer_size, cudaDevAttrL2CacheSize, dev_id));
      if (m_buffer_size > 0) {
        void *buffer = m_l2_buffer;
        CHECK_CUDA(cudaMalloc(&buffer, static_cast<std::size_t>(m_buffer_size)));
        m_l2_buffer = reinterpret_cast<int *>(buffer);
      }
    }
    CudaBackend::L2Flusher::~L2Flusher() {
      if (m_l2_buffer) {
        CHECK_CUDA(cudaFree(m_l2_buffer));
    }
    }

    void CudaBackend::L2Flusher::flush(std::shared_ptr<cudaStream_t> stream) {
      CHECK_CUDA(cudaMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_buffer_size), *stream));
    }
  } // namespace Backend

} // namespace Baseliner
