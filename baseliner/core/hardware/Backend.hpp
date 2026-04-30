#ifndef BASELINER_CORE_HARDWARE_BACKEND_HPP
#define BASELINER_CORE_HARDWARE_BACKEND_HPP
#include <baseliner/cli/Serializer.hpp>
#include <baseliner/core/Durations.hpp>

#include <baseliner/core/Options.hpp>

#include <baseliner/core/Timer.hpp>
#include <iostream>
#include <memory>
#include <thread>
namespace Baseliner::Hardware {
  struct HardwareInfo {
    std::string card_name;
  };
} // namespace Baseliner::Hardware
namespace Baseliner {
  DESCRIBE(Hardware::HardwareInfo, FIELD(card_name))
}

namespace Baseliner::Hardware {

  template <typename BackendT>
  class L2Flusher;

  template <typename BackendT>
  class BlockingKernel;
  template <typename BackendT>
  class GpuTimer;

  template <typename S, typename O>
  class Backend : public IOption {
    friend class BlockingKernel<Backend<S, O>>;
    friend class L2Flusher<Backend<S, O>>;
    friend class GpuTimer<Backend<S, O>>;

  public:
    using stream_t = S;
    using launch_result_t = O;
    static auto instance() -> Backend<S, O> * {
      static Backend<S, O> backend;
      return &backend;
    }

    auto create_stream() -> std::shared_ptr<stream_t> {
      this->set_device();
      return Backend<S, O>::inner_create_stream();
    };
    static void warm_gpu(stream_t stream);
    static void cool_gpu(stream_t & /* stream*/) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    static auto get_device_count() -> int;
    static void synchronize(stream_t stream);
    static void get_last_error();
    void set_device() {
      if (m_device >= Backend<S, O>::get_device_count()) {
        throw Errors::hardware_illegal_device_setting(m_device, Backend<S, O>::get_device_count());
      }
      Backend<S, O>::set_device(m_device);
    };
    auto get_device_info() -> HardwareInfo;
    auto get_current_device() -> int {
      this->set_device();
      return m_device;
    }
    static void reset_device();
    void register_options() override {
      this->add_option("Backend", "device", "The device number to run on", m_device);
      this->add_option("Backend", "lock_clock", "If the clocks should be locked", m_lock_clock);
      this->add_option("Backend", "min_clock_value",
                       "-1 means locking to the base clock, else is locking to a specific frequency",
                       m_min_clock_value);
      this->add_option("Backend", "max_clock_value",
                       "-1 means locking to the base clock, else is locking to a specific frequency",
                       m_max_clock_value);
    }
    void on_update() override {
      this->set_device();
      if (m_lock_clock) {
        Backend<S, O>::lock_clocks(m_min_clock_value, m_max_clock_value);
      } else if (m_was_locked) {
        Backend<S, O>::unlock_clock();
      }
      m_was_locked = m_lock_clock;
    };

  private:
    static void unlock_clock();
    static void lock_clocks(int min_clock_val, int max_clock_val);
    static auto inner_create_stream() -> std::shared_ptr<stream_t>;
    static void set_device(int device);
    Backend() = default;
    int m_device = 0;
    bool m_was_locked = false;
    bool m_lock_clock = false;
    int m_min_clock_value = -1;
    int m_max_clock_value = -1;
  };

  template <typename BackendT>
  class L2Flusher {
  public:
    static auto instance() -> L2Flusher<BackendT> * {
      static L2Flusher<BackendT> flusher;
      return &flusher;
    }
    void flush(typename BackendT::stream_t stream);
    auto operator=(L2Flusher &&) -> L2Flusher & = delete;
    auto operator=(const L2Flusher &) -> L2Flusher & = delete;
    L2Flusher(const L2Flusher &) = delete;
    L2Flusher(L2Flusher &&other) = delete;
    ~L2Flusher() {
      int current_device = BackendT::instance()->get_current_device();
      int max_device = BackendT::get_device_count();
      for (int device = 0; device < max_device; device++) {
        BackendT::set_device(device);
        free(device);
      }
      BackendT::set_device(current_device);
    }

  private:
    void alloc(int device);
    void free(int device);
    L2Flusher() {
      int current_device = BackendT::instance()->get_current_device();
      int max_device = BackendT::get_device_count();
      m_l2_buffer_v.resize(max_device);
      m_buffer_size_v.resize(max_device);
      for (int device = 0; device < max_device; device++) {
        BackendT::set_device(device);
        alloc(device);
      }
      BackendT::set_device(current_device);
    };
    std::vector<int> m_buffer_size_v{}; // NOLINT
    std::vector<int *> m_l2_buffer_v{}; // NOLINT
  };

  template <typename BackendT>
  class BlockingKernel {
  public:
    static auto instance() -> BlockingKernel<BackendT> * {
      static BlockingKernel<BackendT> blocking;
      return &blocking;
    }
    void block(typename BackendT::stream_t stream, double timeout);
    void unblock() {
      int current_device = BackendT::instance()->get_current_device();
      if (m_host_flag_v[current_device] == nullptr || m_host_timeout_flag_v[current_device] == nullptr) {
        return;
      }
      *static_cast<volatile int *>(m_host_flag_v[current_device]) = 1;

      if (*static_cast<volatile int *>(m_host_timeout_flag_v[current_device]) != 0) {
        BlockingKernel::timeout_detected();
      }
    }

    BlockingKernel(BlockingKernel &&other) = delete;
    auto operator=(BlockingKernel &&other) noexcept -> BlockingKernel & = delete;
    BlockingKernel(const BlockingKernel &) = delete;
    auto operator=(const BlockingKernel &) -> BlockingKernel & = delete;

    ~BlockingKernel() {
      int current_device = BackendT::instance()->get_current_device();
      int max_device = BackendT::get_device_count();
      for (int device = 0; device < max_device; device++) {
        BackendT::set_device(device);
        free(device);
      }
      BackendT::set_device(current_device);
    };

  private:
    void alloc(int device);
    void free(int device);
    std::vector<int *> m_host_flag_v;           // NOLINT
    std::vector<int *> m_host_timeout_flag_v;   // NOLINT
    std::vector<int *> m_device_flag_v;         // NOLINT
    std::vector<int *> m_device_timeout_flag_v; // NOLINT

    BlockingKernel() {
      int current_device = BackendT::instance()->get_current_device();
      int max_device = BackendT::get_device_count();
      m_host_flag_v.resize(max_device);
      m_host_timeout_flag_v.resize(max_device);
      m_device_flag_v.resize(max_device);
      m_device_timeout_flag_v.resize(max_device);
      for (int device = 0; device < max_device; device++) {
        BackendT::set_device(device);
        alloc(device);
      }
      BackendT::set_device(current_device);
    };

    static void timeout_detected() {
      std::cout << "Deadlock detected" << "\n";
    };
  };
  template <typename BackendT>
  class GpuTimer : public ITimer<BackendT> {};

} // namespace Baseliner::Hardware

#endif // BACKEND_HPP
