#include <baseliner/backend/cuda/CudaBackend.hpp>
namespace Baseliner {
  namespace Backend {

    template <>
    L2Flusher<CudaBackend>::L2Flusher() {
      int dev_id{};
      CHECK_CUDA(cudaGetDevice(&dev_id));
      CHECK_CUDA(cudaDeviceGetAttribute(&m_buffer_size, cudaDevAttrL2CacheSize, dev_id));
      if (m_buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&m_l2_buffer, static_cast<std::size_t>(m_buffer_size)));
      }
    }
    template <>
    L2Flusher<CudaBackend>::~L2Flusher() {
      if (m_l2_buffer != nullptr) {
        CHECK_CUDA(cudaFree(m_l2_buffer));
      }
    }

    template <>
    void L2Flusher<CudaBackend>::flush(std::shared_ptr<typename CudaBackend::stream_t> stream) {
      if (m_l2_buffer != nullptr) {
        CHECK_CUDA(cudaMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_buffer_size), *stream));
      }
    }
  } // namespace Backend

} // namespace Baseliner
