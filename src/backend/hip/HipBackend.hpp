#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP
#include "backend/Backend.hpp"
#include "Kernel.hpp"
#include <hip/hip_runtime.h>
void check_hip_error(hipError_t error_code, const char *file, int line);
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)

namespace Baseliner {
  using IHipKernel = IKernel<hipStream_t>;
  namespace Backend {
    class HipBackend : public IDevice<hipEvent_t, hipStream_t> {
    public:
      void synchronize(hipStream_t stream) override;
      void set_device(int device) override;
      void reset_device() override;
      class L2Flusher : public IDevice::L2Flusher {
      public:
        L2Flusher();
        ~L2Flusher();
        void flush(hipStream_t stream) override;
      };
      class BlockingKernel : public IDevice::BlockingKernel {
      public:
        BlockingKernel();
        void block(hipStream_t stream, double timeout) override;
        ~BlockingKernel();
      };
      class GpuTimer : public IDevice::GpuTimer {
      public:
        GpuTimer();
        ~GpuTimer();
        void start(hipStream_t stream) override;
        void stop(hipStream_t stream) override;
        float time_elapsed() override;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // HIP_BACKEND_HPP