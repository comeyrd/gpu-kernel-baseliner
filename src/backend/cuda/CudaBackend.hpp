#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include "backend/Backend.hpp"
void check_cuda_error(cudaError_t error_code, const char *file, int line);
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

namespace Baseliner {
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
        float time_elapsed();

      private:
        cudaEvent_t start_event;
        cudaEvent_t stop_event;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // CUDA_BACKEND_HPP