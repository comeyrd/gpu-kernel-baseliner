#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include "ITimer.hpp"
#include "Kernel.hpp"
#include "backend/Backend.hpp"
#include <chrono>
void check_cuda_error(cudaError_t error_code, const char *file, int line);
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__)

namespace Baseliner {
  template <typename Input, typename Output>
  using ICudaKernel = IKernel<cudaStream_t, Input, Output>;
  namespace Backend {
    class CudaBackend : public IDevice<cudaEvent_t, cudaStream_t> {
    public:
      std::shared_ptr<cudaStream_t> create_stream() override;
      void synchronize(std::shared_ptr<cudaStream_t> stream) override;
      void set_device(int device) override;
      void reset_device() override;
      class L2Flusher : public IDevice::L2Flusher {
      public:
        L2Flusher();
        ~L2Flusher();
        void flush(std::shared_ptr<cudaStream_t> stream) override;
      };
      class BlockingKernel : public IDevice::BlockingKernel {
      public:
        BlockingKernel();
        void block(std::shared_ptr<cudaStream_t> stream, double timeout) override;
        ~BlockingKernel();
      };
      class GpuTimer : public IDevice::GpuTimer {
      public:
        GpuTimer(std::shared_ptr<cudaStream_t> stream)
            : IDevice::GpuTimer(stream) {
          CHECK_CUDA(cudaEventCreate(&m_start_event));
          CHECK_CUDA(cudaEventCreate(&m_stop_event));
        };
        ~GpuTimer() {
          CHECK_CUDA(cudaEventDestroy(m_start_event));
          CHECK_CUDA(cudaEventDestroy(m_stop_event));
        };
        void start() override;
        void stop() override;
        float_milliseconds time_elapsed() override;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // CUDA_BACKEND_HPP