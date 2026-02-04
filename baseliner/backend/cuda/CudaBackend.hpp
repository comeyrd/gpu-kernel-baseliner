#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include <baseliner/Kernel.hpp>
#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>

void check_cuda_error(cudaError_t error_code, const char *file, int line);
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__)

namespace Baseliner {
  template <typename Input, typename Output>
  class ICudaKernel : public IKernel<cudaStream_t, Input, Output> {
  public:
    ICudaKernel(const Input &input)
        : IKernel<cudaStream_t, Input, Output>(input) {
      CHECK_CUDA(cudaEventCreate(&m_start_event));
      CHECK_CUDA(cudaEventCreate(&m_stop_event));
    }
    ~ICudaKernel() {
      CHECK_CUDA(cudaEventDestroy(m_start_event));
      CHECK_CUDA(cudaEventDestroy(m_stop_event));
    }

    void measure_start(std::shared_ptr<cudaStream_t> &stream) override final {
      CHECK_CUDA(cudaEventRecord(m_start_event, *stream));
    };
    void measure_stop(std::shared_ptr<cudaStream_t> &stream) override final {
      CHECK_CUDA(cudaEventRecord(m_stop_event, *stream));
    };
    float_milliseconds time_elapsed() final {
      float result;
      CHECK_CUDA(cudaEventSynchronize(m_stop_event));
      CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    };

  private:
    cudaEvent_t m_start_event;
    cudaEvent_t m_stop_event;
  };

  namespace Backend {
    class CudaBackend : public IDevice<cudaStream_t> {
    public:
      std::shared_ptr<cudaStream_t> create_stream() override;
      void get_last_error() override;
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
      class Timer : public IDevice::Timer {
      public:
        Timer()
            : IDevice::Timer() {
          CHECK_CUDA(cudaEventCreate(&m_start_event));
          CHECK_CUDA(cudaEventCreate(&m_stop_event));
        }
        ~Timer() {
          CHECK_CUDA(cudaEventDestroy(m_start_event));
          CHECK_CUDA(cudaEventDestroy(m_stop_event));
        }
        void measure_start(std::shared_ptr<cudaStream_t> &stream) override final {
          CHECK_CUDA(cudaEventRecord(m_start_event, *stream));
        };
        void measure_stop(std::shared_ptr<cudaStream_t> &stream) override final {
          CHECK_CUDA(cudaEventRecord(m_stop_event, *stream));
        };
        float_milliseconds time_elapsed() final {
          float result;
          CHECK_CUDA(cudaEventSynchronize(m_stop_event));
          CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
          return float_milliseconds(result);
        };

      private:
        cudaEvent_t m_start_event;
        cudaEvent_t m_stop_event;
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // CUDA_BACKEND_HPP