#ifndef BASELINER_CORE_HARDWARE_CUDA_CUDABACKEND_HPP
#define BASELINER_CORE_HARDWARE_CUDA_CUDABACKEND_HPP
#include "cuda_runtime.h"
#include <baseliner/core/Benchmark.hpp>

#include <baseliner/core/Timer.hpp>
#include <baseliner/core/hardware/Backend.hpp>

void check_cuda_error(cudaError_t error_code, const char *file, int line);                // NOLINT
void check_cuda_error_no_except(cudaError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_CUDA_NO_EXCEPT(error) check_cuda_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Hardware {
    using CudaBackend = Backend<cudaStream_t, std::monostate>;
    template <>
    class GpuTimer<CudaBackend> : public ITimer<CudaBackend> {
    public:
      using Stream = ITimer<CudaBackend>::Stream;
      using Workload = ITimer<CudaBackend>::Workload;
      using Funct = ITimer<CudaBackend>::Funct;

      ~GpuTimer() = default;

      void init(Stream &stream) override {
        if (m_state != State::Idle) {
          throw Errors::timer_init_on_not_idle();
        }
        reset();
        m_state = State::Single;
        m_starts.resize(1);
        m_stops.resize(1);
        CHECK_CUDA(cudaEventCreate(&m_starts[0]));
        CHECK_CUDA(cudaEventCreate(&m_stops[0]));
      }

      void measure_before(Stream &stream) override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("measure_before()");
        }
        CHECK_CUDA(cudaEventRecord(m_starts[0], stream));
      }

      // No-op for CUDA: timing is driven by cudaEvent_t, not launch_result_t
      void measure_consume(typename CudaBackend::launch_result_t event) override {
      }

      void measure_after(Stream &stream) override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("measure_after()");
        }
        CHECK_CUDA(cudaEventRecord(m_stops[0], stream));
      }

      auto elapsed() -> float_milliseconds override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("elapsed()");
        }
        CHECK_CUDA(cudaEventSynchronize(m_stops[0]));
        float temp_f{};
        CHECK_CUDA(cudaEventElapsedTime(&temp_f, m_starts[0], m_stops[0]));
        m_state = State::Idle;
        return float_milliseconds(temp_f);
      }

      void init_batch(Stream &stream, size_t batch_size, bool is_blocking) override {
        if (m_state != State::Idle) {
          throw Errors::timer_init_on_not_idle();
        }
        reset();
        m_state = State::Batch;
        m_is_blocking = is_blocking;
        m_batch_size = batch_size;
        m_starts.resize(batch_size);
        m_stops.resize(batch_size);
        for (size_t idx = 0; idx < batch_size; idx++) {
          CHECK_CUDA(cudaEventCreate(&m_starts[idx]));
          CHECK_CUDA(cudaEventCreate(&m_stops[idx]));
        }
      }

      void measure_batch_before(Stream &stream) override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("measure_batch_before()");
        }
        if (m_pos_batch >= m_batch_size) {
          throw Errors::timer_more_measure_than_batch(m_batch_size);
        }
        CHECK_CUDA(cudaEventRecord(m_starts[m_pos_batch], stream));
      }

      // No-op for CUDA
      void measure_batch_consume(typename CudaBackend::launch_result_t event) override {
      }

      void measure_batch_after(Stream &stream) override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("measure_batch_after()");
        }
        CHECK_CUDA(cudaEventRecord(m_stops[m_pos_batch], stream));
        m_pos_batch++;
      }

      auto elapsed_batch() -> std::vector<float_milliseconds> override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("elapsed_batch()");
        }
        std::vector<float_milliseconds> result_vec;
        result_vec.reserve(m_pos_batch);
        CHECK_CUDA(cudaEventSynchronize(m_stops[m_pos_batch - 1]));
        for (size_t idx = 0; idx < m_pos_batch; idx++) {
          float temp_f{};
          CHECK_CUDA(cudaEventElapsedTime(&temp_f, m_starts[idx], m_stops[idx]));
          result_vec.emplace_back(temp_f);
        }
        m_state = State::Idle;
        return result_vec;
      }

    private:
      void reset() {
        for (auto start : m_starts) {
          CHECK_CUDA(cudaEventDestroy(start));
        }
        for (auto stop : m_stops) {
          CHECK_CUDA(cudaEventDestroy(stop));
        }
        m_starts.clear();
        m_stops.clear();
        m_batch_size = 0;
        m_pos_batch = 0;
      }

      enum class State : char {
        Idle,
        Single,
        Batch
      };

      size_t m_pos_batch = 0;
      State m_state = State::Idle;
      bool m_is_blocking{false};
      size_t m_batch_size = 0;
      std::vector<cudaEvent_t> m_starts;
      std::vector<cudaEvent_t> m_stops;
    };
  } // namespace Hardware
  using ICudaWorkload = IWorkload<Hardware::CudaBackend>;
  using CudaBenchmark = Benchmark<Hardware::CudaBackend>;

} // namespace Baseliner

#ifdef BASELINER_HAS_NVML
#include <nvml.h>
void check_nvml_error(nvmlReturn_t error_code, const char *file, int line);               // NOLINT
void check_nvml_error_no_except(nvmlReturn_t error_code, const char *file, int line);     // NOLINT
#define CHECK_NVML(error) check_nvml_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_NVML_NO_EXCEPT(error) check_nvml_error_no_except(error, __FILE__, __LINE__) // NOLINT
class NvmlManager {
public:
  // This is called automatically the first time Instance() is accessed
  NvmlManager() {
    nvmlInit();
  }

  // This is called when the program exits
  ~NvmlManager() {
    nvmlShutdown();
  }
  static auto get_current_device() -> nvmlDevice_t {
    ensure_init();

    int cudaIdx = 0;
    CHECK_CUDA(cudaGetDevice(&cudaIdx));

    char pciBusId[64];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pciBusId, 64, cudaIdx));

    // 3. Ask NVML for the handle matching that specific PCI Bus ID
    nvmlDevice_t Backend;
    CHECK_NVML(nvmlDeviceGetHandleByPciBusId(pciBusId, &Backend));
    return Backend;
  }

  static void ensure_init() {
    static NvmlManager instance;
  }
};

#endif
#endif // CUDA_BACKEND_HPP