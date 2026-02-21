#ifndef BACKEND_HPP
#define BACKEND_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Timer.hpp>
#include <iostream>
#include <memory>
namespace Baseliner::Backend {
  template <typename S>
  class Backend {
  public:
    using stream_t = S;
    auto create_stream() -> std::shared_ptr<stream_t>;
    void synchronize(std::shared_ptr<stream_t> stream);
    void get_last_error();
    void set_device(int Backend);
    void reset_device();
    ~Backend() = default;
  };

  template <typename BackendT>
  class L2Flusher {
  public:
    void flush(std::shared_ptr<typename BackendT::stream_t> stream);
    ~L2Flusher();
    L2Flusher();
    L2Flusher(L2Flusher &&other) noexcept
        : m_buffer_size(other.m_buffer_size),
          m_l2_buffer(other.m_l2_buffer) {
      other.m_l2_buffer = nullptr;
    }

    auto operator=(L2Flusher &&other) noexcept -> L2Flusher & {
      if (this != &other) {
        m_buffer_size = other.m_buffer_size;
        m_l2_buffer = other.m_l2_buffer;

        other.m_l2_buffer = nullptr;
      }
      return *this;
    }

  protected:
    int m_buffer_size{}; // NOLINT
    int *m_l2_buffer{};  // NOLINT
  };
  template <typename BackendT>
  class BlockingKernel {
  public:
    void block(std::shared_ptr<typename BackendT::stream_t> stream, double timeout);
    void unblock() {
      if (m_host_flag == nullptr || m_host_timeout_flag == nullptr) {
        return;
      }
      *static_cast<volatile int *>(m_host_flag) = 1;

      if (*static_cast<volatile int *>(m_host_timeout_flag) != 0) {
        BlockingKernel::timeout_detected();
      }
    }

    BlockingKernel(BlockingKernel &&other) noexcept
        : m_host_flag(other.m_host_flag),
          m_host_timeout_flag(other.m_host_timeout_flag),
          m_device_flag(other.m_device_flag),
          m_device_timeout_flag(other.m_device_timeout_flag) {
      other.m_device_flag = nullptr;
      other.m_device_timeout_flag = nullptr;
      other.m_host_flag = nullptr;
      other.m_host_timeout_flag = nullptr;
    }
    ~BlockingKernel();
    BlockingKernel();
    auto operator=(BlockingKernel &&other) noexcept -> BlockingKernel &;

    BlockingKernel(const BlockingKernel &) = delete;
    auto operator=(const BlockingKernel &) -> BlockingKernel & = delete;

  protected:
    int *m_host_flag;             // NOLINT
    int *m_host_timeout_flag;     // NOLINT
    int *m_device_flag{};         // NOLINT
    int *m_device_timeout_flag{}; // NOLINT

    static void timeout_detected() {
      std::cout << "Deadlock detected" << "\n";
    };
  };

  template <typename BackendT>
  class GpuTimer {
  public:
    ~GpuTimer();
    GpuTimer();
    GpuTimer(const GpuTimer &) = delete;
    auto operator=(const GpuTimer &) -> GpuTimer & = delete;
    GpuTimer(GpuTimer &&) = delete;
    auto operator=(GpuTimer &&) -> GpuTimer & = delete;
    auto time_elapsed() -> float_milliseconds;

    void measure_start(std::shared_ptr<typename BackendT::stream_t> stream);
    void measure_stop(std::shared_ptr<typename BackendT::stream_t> stream);

  protected:
  };
} // namespace Baseliner::Backend

#endif // BACKEND_HPP
