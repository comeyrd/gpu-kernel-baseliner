#ifndef CUDA_BACKEND_HPP
#define CUDA_BACKEND_HPP
#include "baseliner/Kernel.hpp"
#include "cuda_runtime.h"
#include <baseliner/backend/Backend.hpp>
void check_cuda_error(cudaError_t error_code, const char *file, int line);                // NOLINT
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  template <typename Input, typename Output>
  class ICudaKernel : public IKernel<cudaStream_t, Input, Output> {
  public:
    ICudaKernel(const std::shared_ptr<const Input> input)
        : IKernel<cudaStream_t, Input, Output>(input) {
      CHECK_CUDA(cudaEventCreate(&m_start_event));
      CHECK_CUDA(cudaEventCreate(&m_stop_event));
    }
    ~ICudaKernel() override { // NOLINT
      CHECK_CUDA(cudaEventDestroy(m_start_event));
      CHECK_CUDA(cudaEventDestroy(m_stop_event));
    }

    void measure_start(std::shared_ptr<cudaStream_t> stream) final {
      CHECK_CUDA(cudaEventRecord(m_start_event, *stream));
    };
    void measure_stop(std::shared_ptr<cudaStream_t> stream) final {
      CHECK_CUDA(cudaEventRecord(m_stop_event, *stream));
    };
    auto time_elapsed() -> float_milliseconds final {
      float result{};
      CHECK_CUDA(cudaEventSynchronize(m_stop_event));
      CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
      return float_milliseconds(result);
    };

    // No Copy or Moving
    ICudaKernel(const ICudaKernel &) = delete;
    auto operator=(const ICudaKernel &) -> ICudaKernel & = delete;
    ICudaKernel(ICudaKernel &&) = delete;
    auto operator=(ICudaKernel &&) -> ICudaKernel & = delete;

  private:
    cudaEvent_t m_start_event{};
    cudaEvent_t m_stop_event{};
  };

  namespace Backend {
    class CudaBackend : public IDevice<cudaStream_t> {
    public:
      auto create_stream() -> std::shared_ptr<cudaStream_t> override;
      void get_last_error() override;
      void synchronize(std::shared_ptr<cudaStream_t> stream) override;
      void set_device(int device) override;
      void reset_device() override;
      CudaBackend() {
        set_device(2);
      };
      class L2Flusher : public IDevice::IL2Flusher {
      public:
        L2Flusher();
        ~L2Flusher() override;
        void flush(std::shared_ptr<cudaStream_t> stream) override;
      };
      class BlockingKernel : public IDevice::IBlockingKernel {
      public:
        BlockingKernel();
        void block(std::shared_ptr<cudaStream_t> stream, double timeout) override;
        ~BlockingKernel() override;
      };
      class Timer : public IDevice::ITimer { // NOLINT
      public:
        Timer() // NOLINT
            : IDevice::ITimer() {
          CHECK_CUDA(cudaEventCreate(&m_start_event));
          CHECK_CUDA(cudaEventCreate(&m_stop_event));
        }
        ~Timer() override {
          CHECK_CUDA(cudaEventDestroy(m_start_event));
          CHECK_CUDA(cudaEventDestroy(m_stop_event));
        }
        void measure_start(std::shared_ptr<cudaStream_t> stream) final {
          CHECK_CUDA(cudaEventRecord(m_start_event, *stream));
        };
        void measure_stop(std::shared_ptr<cudaStream_t> stream) final {
          CHECK_CUDA(cudaEventRecord(m_stop_event, *stream));
        };
        auto time_elapsed() -> float_milliseconds final {
          float result{};
          CHECK_CUDA(cudaEventSynchronize(m_stop_event));
          CHECK_CUDA(cudaEventElapsedTime(&result, m_start_event, m_stop_event));
          return float_milliseconds(result);
        };

      private:
        cudaEvent_t m_start_event{};
        cudaEvent_t m_stop_event{};
      };
    };

  } // namespace Backend

} // namespace Baseliner

#endif // CUDA_BACKEND_HPP