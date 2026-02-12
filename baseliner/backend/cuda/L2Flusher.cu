#include <baseliner/backend/cuda/CudaBackend.hpp>
namespace Baseliner {
  namespace Backend {

    CudaBackend::L2Flusher::L2Flusher() {
      int dev_id{};
      CHECK_CUDA(cudaGetDevice(&dev_id));
      CHECK_CUDA(cudaDeviceGetAttribute(&m_buffer_size, cudaDevAttrL2CacheSize, dev_id));
      if (m_buffer_size > 0) {
        CHECK_CUDA(cudaMalloc(&m_l2_buffer, static_cast<std::size_t>(m_buffer_size)));
      }
    }
    CudaBackend::L2Flusher::~L2Flusher() {
      if (m_l2_buffer != nullptr) {
        CHECK_CUDA(cudaFree(m_l2_buffer));
      }
    }

    void CudaBackend::L2Flusher::flush(std::shared_ptr<cudaStream_t> stream) {
      if (m_l2_buffer != nullptr) {
        CHECK_CUDA(cudaMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_buffer_size), *stream));
      }
    }
    CudaBackend::L2Flusher::L2Flusher(L2Flusher &&other) noexcept
        : IL2Flusher(std::move(other)) {
    }

    auto CudaBackend::L2Flusher::L2Flusher::operator=(L2Flusher &&other) noexcept -> CudaBackend::L2Flusher & {
      if (this != &other) {
        IL2Flusher::operator=(std::move(other));
      }
      return *this;
    }
  } // namespace Backend

} // namespace Baseliner
