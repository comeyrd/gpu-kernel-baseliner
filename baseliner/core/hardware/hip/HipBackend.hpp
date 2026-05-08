#ifndef BASELINER_CORE_HARDWARE_HIP_HIPBACKEND_HPP
#define BASELINER_CORE_HARDWARE_HIP_HIPBACKEND_HPP
#include "hip/hip_runtime.h"
#include <baseliner/core/Benchmark.hpp>

#include <baseliner/core/hardware/Backend.hpp>

void check_hip_error(hipError_t error_code, const char *file, int line);                // NOLINT
void check_hip_error_no_except(hipError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Hardware {
    using HipBackend = Backend<hipStream_t, std::monostate>;
    template <>
    class GpuTimer<HipBackend> : public ITimer<HipBackend> {
    public:
      using Stream = ITimer<HipBackend>::Stream;

      ~GpuTimer() = default;

      void init(Stream /*stream*/) override {
        if (m_state != State::Idle) {
          throw Errors::timer_init_on_not_idle();
        }
        reset();
        m_state = State::Single;
        m_starts.resize(1);
        m_stops.resize(1);
        CHECK_HIP(hipEventCreate(&m_starts[0]));
        CHECK_HIP(hipEventCreate(&m_stops[0]));
      }

      void measure_before(Stream stream) override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("measure_before()");
        }
        CHECK_HIP(hipEventRecord(m_starts[0], stream));
      }

      // No-op for HIP: timing is driven by hipEvent_t, not launch_result_t
      void measure_consume(typename HipBackend::launch_result_t /*event*/) override {
      }

      void measure_after(Stream stream) override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("measure_after()");
        }
        CHECK_HIP(hipEventRecord(m_stops[0], stream));
      }

      auto elapsed() -> float_milliseconds override {
        if (m_state != State::Single) {
          throw Errors::timer_not_single_state("elapsed()");
        }
        CHECK_HIP(hipEventSynchronize(m_stops[0]));
        float temp_f{};
        CHECK_HIP(hipEventElapsedTime(&temp_f, m_starts[0], m_stops[0]));
        m_state = State::Idle;
        return float_milliseconds(temp_f);
      }

      void init_batch(Stream /*stream*/, size_t batch_size, bool is_blocking) override {
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
          CHECK_HIP(hipEventCreate(&m_starts[idx]));
          CHECK_HIP(hipEventCreate(&m_stops[idx]));
        }
      }

      void measure_batch_before(Stream stream) override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("measure_batch_before()");
        }
        if (m_pos_batch >= m_batch_size) {
          throw Errors::timer_more_measure_than_batch(m_batch_size);
        }
        CHECK_HIP(hipEventRecord(m_starts[m_pos_batch], stream));
      }

      // No-op for HIP
      void measure_batch_consume(typename HipBackend::launch_result_t /*event*/) override {
      }

      void measure_batch_after(Stream stream) override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("measure_batch_after()");
        }
        CHECK_HIP(hipEventRecord(m_stops[m_pos_batch], stream));
        m_pos_batch++;
      }

      auto elapsed_batch() -> std::vector<float_milliseconds> override {
        if (m_state != State::Batch) {
          throw Errors::timer_not_batch("elapsed_batch()");
        }
        std::vector<float_milliseconds> result_vec;
        result_vec.reserve(m_pos_batch);
        CHECK_HIP(hipEventSynchronize(m_stops[m_pos_batch - 1]));
        for (size_t idx = 0; idx < m_pos_batch; idx++) {
          float temp_f{};
          CHECK_HIP(hipEventElapsedTime(&temp_f, m_starts[idx], m_stops[idx]));
          result_vec.emplace_back(temp_f);
        }
        m_state = State::Idle;
        return result_vec;
      }

    private:
      void reset() {
        for (hipEvent_t &start : m_starts) {
          CHECK_HIP(hipEventDestroy(start));
        }
        for (hipEvent_t &stop : m_stops) {
          CHECK_HIP(hipEventDestroy(stop));
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
      std::vector<hipEvent_t> m_starts;
      std::vector<hipEvent_t> m_stops;
    };

  } // namespace Hardware
  using IHipWorkload = IWorkload<Hardware::HipBackend>;
  using HipBenchmark = Benchmark<Hardware::HipBackend>;

} // namespace Baseliner

#ifdef BASELINER_HAS_AMDSMI
#include "amd_smi/amdsmi.h"
void check_amd_smi_error(amdsmi_status_t error_code, const char *file, int line);              // NOLINT
void check_amd_smi_error_no_except(amdsmi_status_t error_code, const char *file, int line);    // NOLINT
#define CHECK_AMDSMI(error) check_amd_smi_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_AMDSMI_NO_EXCEPT(error) check_amd_smi_error_no_except(error, __FILE__, __LINE__) // NOLINT
class AmdSmiManager {
public:
  // This is called automatically the first time Instance() is accessed
  AmdSmiManager() {
    CHECK_AMDSMI(amdsmi_init(AMDSMI_INIT_AMD_GPUS));
    uint32_t socket_count = 0;
    CHECK_AMDSMI(amdsmi_get_socket_handles(&socket_count, nullptr));
    m_sockets.resize(socket_count);
    CHECK_AMDSMI(amdsmi_get_socket_handles(&socket_count, m_sockets.data()));
    uint32_t global_gpu_index = 0;
    for (auto socket : m_sockets) {
      uint32_t processor_count = 0;
      CHECK_AMDSMI(amdsmi_get_processor_handles(socket, &processor_count, nullptr));
      m_processors.resize(processor_count);
      CHECK_AMDSMI(amdsmi_get_processor_handles(socket, &processor_count, m_processors.data()));
    }
  }

  AmdSmiManager(const AmdSmiManager &) = delete;
  AmdSmiManager(AmdSmiManager &&) = delete;
  AmdSmiManager &operator=(const AmdSmiManager &) = delete;
  AmdSmiManager &operator=(AmdSmiManager &&) = delete;
  // This is called when the program exits
  ~AmdSmiManager() {
    CHECK_AMDSMI(amdsmi_shut_down());
  }
  auto get_current_device() const -> amdsmi_processor_handle {
    int hip_device_id = 0;
    CHECK_HIP(hipGetDevice(&hip_device_id));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, hip_device_id));
    amdsmi_bdf_t bdf = {};
    bdf.domain_number = static_cast<uint16_t>(prop.pciDomainID);
    bdf.bus_number = static_cast<uint8_t>(prop.pciBusID);
    bdf.device_number = static_cast<uint8_t>(prop.pciDeviceID);
    bdf.function_number = 0; // Standard for primary GPU handles

    amdsmi_processor_handle proc;
    CHECK_AMDSMI(amdsmi_get_processor_handle_from_bdf(bdf, &proc));

    return proc;
  }

  static auto ensure_init() -> AmdSmiManager * {
    static AmdSmiManager instance;
    return &instance;
  }

private:
  std::vector<amdsmi_socket_handle> m_sockets;
  std::vector<amdsmi_processor_handle> m_processors;
};
#endif // BASELINER_HAS_AMDSMI
#endif // HIP_BACKEND_HPP
