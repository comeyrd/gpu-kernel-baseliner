#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include <chrono>
#include "Kernel.hpp"
#include "backend/Backend.hpp"
void check_cuda_error(cudaError_t error_code, const char *file, int line);
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

namespace Baseliner {
  using ICudaKernel = IKernel<cudaStream_t>;

  namespace Backend {
    class CudaBackend : public IDevice<cudaEvent_t, cudaStream_t> {
    public:
      void synchronize(cudaStream_t stream) override;
      void set_device(int device) override;
      void reset_device() override;
      class L2Flusher : public IDevice::L2Flusher {
      public:
        L2Flusher();
        ~L2Flusher();
        void flush(cudaStream_t stream) override;
      };
      class BlockingKernel : public IDevice::BlockingKernel {
      public:
        BlockingKernel();
        void block(cudaStream_t stream, double timeout) override;
        ~BlockingKernel();
      };
      class GpuTimer : public IDevice::GpuTimer {
      public:
        GpuTimer();
        ~GpuTimer();
        void start(cudaStream_t stream) override;
        void stop(cudaStream_t stream) override;
        std::chrono::duration<float, std::milli> time_elapsed() override;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // CUDA_BACKEND_HPP