#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP
#include "hip/hip_runtime.h"
#include <baseliner/Kernel.hpp>
#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>

void check_hip_error(hipError_t error_code, const char *file, int line);
void check_hip_error_no_except(hipError_t error_code, const char *file, int line);
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__)

namespace Baseliner {
  template <typename Input, typename Output>
  class IHipKernel : public IKernel<hipStream_t, Input, Output> {
  public:
    IHipKernel(const Input &input)
        : IKernel<hipStream_t, Input, Output>(input) {
      CHECK_HIP(hipEventCreate(&m_start_event));
      CHECK_HIP(hipEventCreate(&m_stop_event));
    }
    ~IHipKernel() {
      CHECK_HIP(hipEventDestroy(m_start_event));
      CHECK_HIP(hipEventDestroy(m_stop_event));
    }

    void measure_start(std::shared_ptr<hipStream_t> &stream) override final {
      CHECK_HIP(hipEventRecord(m_start_event, *stream));
    };
    void measure_stop(std::shared_ptr<hipStream_t> &stream) override final {
      CHECK_HIP(hipEventRecord(m_stop_event, *stream));
    };
    float_milliseconds time_elapsed() final {
      float result;
      CHECK_HIP(hipEventSynchronize(m_stop_event));
      CHECK_HIP(hipEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    };

  private:
    hipEvent_t m_start_event;
    hipEvent_t m_stop_event;
  };

  namespace Backend {
    class HipBackend : public IDevice<hipStream_t> {
    public:
      std::shared_ptr<hipStream_t> create_stream() override;
      void get_last_error() override;
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
      class Timer : public IDevice::Timer {
      public:
        Timer()
            : IDevice::Timer() {
          CHECK_HIP(hipEventCreate(&m_start_event));
          CHECK_HIP(hipEventCreate(&m_stop_event));
        }
        ~Timer() {
          CHECK_HIP(hipEventDestroy(m_start_event));
          CHECK_HIP(hipEventDestroy(m_stop_event));
        }
        void measure_start(std::shared_ptr<hipStream_t> &stream) override final {
          CHECK_HIP(hipEventRecord(m_start_event, *stream));
        };
        void measure_stop(std::shared_ptr<hipStream_t> &stream) override final {
          CHECK_HIP(hipEventRecord(m_stop_event, *stream));
        };
        float_milliseconds time_elapsed() final {
          float result;
          CHECK_HIP(hipEventSynchronize(m_stop_event));
          CHECK_HIP(hipEventElapsedTime(&result, m_start_event, m_stop_event));
          return float_milliseconds(result);
        };

      private:
        hipEvent_t m_start_event;
        hipEvent_t m_stop_event;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // HIP_BACKEND_HPP
