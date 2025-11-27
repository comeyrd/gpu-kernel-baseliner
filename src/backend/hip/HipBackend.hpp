#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP
#include "ITimer.hpp"
#include "Kernel.hpp"
#include "backend/Backend.hpp"
#include <chrono>
#include "hip/hip_runtime.h"
void check_hip_error(hipError_t error_code, const char *file, int line);
void check_hip_error_no_except(hipError_t error_code, const char *file, int line);
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__)

namespace Baseliner {
  template <typename Input, typename Output>
  using IHipKernel = IKernel<hipStream_t, Input, Output>;
  namespace Backend {
    class HipBackend : public IDevice<hipEvent_t, hipStream_t> {
    public:
      std::shared_ptr<hipStream_t> create_stream() override;
      void synchronize(std::shared_ptr<hipStream_t> stream) override;
      void set_device(int device) override;
      void reset_device() override;
      class L2Flusher : public IDevice::L2Flusher {
      public:
        L2Flusher();
        ~L2Flusher();
        void flush(std::shared_ptr<hipStream_t> stream) override;
      };
      class BlockingKernel : public IDevice::BlockingKernel {
      public:
        BlockingKernel();
        void block(std::shared_ptr<hipStream_t> stream, double timeout) override;
        ~BlockingKernel();
      };
      class GpuTimer : public IDevice::GpuTimer {
      public:
        GpuTimer(std::shared_ptr<hipStream_t> stream)
            : IDevice::GpuTimer(stream) {
          CHECK_HIP(hipEventCreate(&m_start_event));
          CHECK_HIP(hipEventCreate(&m_stop_event));
        };
        ~GpuTimer() {
          CHECK_HIP(hipEventDestroy(m_start_event));
          CHECK_HIP(hipEventDestroy(m_stop_event));
        };
        void start() override;
        void stop() override;
        float_milliseconds time_elapsed() override;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // HIP_BACKEND_HPP
