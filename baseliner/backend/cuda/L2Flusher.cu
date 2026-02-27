#include <baseliner/backend/cuda/CudaBackend.hpp>
namespace Baseliner {
  namespace Backend {

    template <>
    void L2Flusher<CudaBackend>::alloc(int device) {
      CHECK_CUDA(cudaDeviceGetAttribute(&m_buffer_size_v[device], cudaDevAttrL2CacheSize, device));
      if (m_buffer_size_v[device] > 0) {
        CHECK_CUDA(cudaMalloc(&m_l2_buffer_v[device], static_cast<std::size_t>(m_buffer_size_v[device])));
      }
    }
    template <>
    void L2Flusher<CudaBackend>::free(int device) {
      CHECK_CUDA(cudaFree(m_l2_buffer_v[device]));
    }

    template <>
    void L2Flusher<CudaBackend>::flush(std::shared_ptr<typename CudaBackend::stream_t> stream) {
      int current_device = CudaBackend::instance()->get_current_device();
      CHECK_CUDA(cudaMemsetAsync(m_l2_buffer_v[current_device], 0,
                                 static_cast<std::size_t>(m_buffer_size_v[current_device]), *stream));
    }
  } // namespace Backend

} // namespace Baseliner
