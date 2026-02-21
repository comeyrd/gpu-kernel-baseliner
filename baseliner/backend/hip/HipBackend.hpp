#ifndef HIP_BACKEND_HPP
#define HIP_BACKEND_HPP
#include "baseliner/Kernel.hpp"
#include "hip/hip_runtime.h"
#include <algorithm>
#include <baseliner/Benchmark.hpp>
#include <baseliner/backend/Backend.hpp>
#include <iterator>

void check_hip_error(hipError_t error_code, const char *file, int line);                // NOLINT
void check_hip_error_no_except(hipError_t error_code, const char *file, int line);      // NOLINT
#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)                     // NOLINT
#define CHECK_HIP_NO_EXCEPT(error) check_hip_error_no_except(error, __FILE__, __LINE__) // NOLINT

namespace Baseliner {
  namespace Backend {
    using HipBackend = Backend<hipStream_t>;
    template <>
    class GpuTimer<HipBackend> {
    public:
      ~GpuTimer();
      GpuTimer();
      GpuTimer(const GpuTimer &) = delete;
      auto operator=(const GpuTimer &) -> GpuTimer & = delete;
      GpuTimer(GpuTimer &&) = delete;
      auto operator=(GpuTimer &&) -> GpuTimer & = delete;

      void measure_start(std::shared_ptr<typename HipBackend::stream_t> stream);
      void measure_stop(std::shared_ptr<typename HipBackend::stream_t> stream);
      auto time_elapsed() -> float_milliseconds;

    protected:
    private:
      hipEvent_t m_start_event{};
      hipEvent_t m_stop_event{};
    };
  } // namespace Backend
  using IHipCase = ICase<Backend::HipBackend>;
  using HipBenchmark = Benchmark<Backend::HipBackend>;

  template <typename Input, typename Output>
  using IHipKernel = IKernel<Backend::HipBackend, Input, Output>;
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

    return m_processors[0];
    // return proc;
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
