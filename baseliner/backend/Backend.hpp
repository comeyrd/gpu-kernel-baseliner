#ifndef BACKEND_HPP
#define BACKEND_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Timer.hpp>
#include <iostream>
#include <memory>
namespace Baseliner::Backend {
  template <typename BackendT>
  class L2Flusher;

  template <typename BackendT>
  class BlockingKernel;
  template <typename BackendT>
  class GpuTimer;

  template <typename S>
  class Backend : public IOption {
    friend class BlockingKernel<Backend<S>>;
    friend class L2Flusher<Backend<S>>;
    friend class GpuTimer<Backend<S>>;

  public:
    using stream_t = S;
    static auto instance() -> Backend<S> * {
      static Backend<S> backend;
      return &backend;
    }

    auto create_stream() -> std::shared_ptr<stream_t> {
      this->set_device();
      return Backend<S>::inner_create_stream();
    };
    static auto get_device_count() -> int;
    static void synchronize(std::shared_ptr<stream_t> stream);
    static void get_last_error();
    void set_device() {
      if (m_device >= Backend<S>::get_device_count()) {
        throw std::runtime_error("Baseliner Error : Trying to set the device to " +
                                 Conversion::baseliner_to_string(m_device) + " but there is only " +
                                 Conversion::baseliner_to_string(Backend<S>::get_device_count()) +
                                 " devices available");
      }
      Backend<S>::set_device(m_device);
    };
    auto get_current_device() -> int {
      this->set_device();
      return m_device;
    }
    static void reset_device();
    void register_options() override {
      this->add_option("Backend", "device", "The device number to run on", m_device);
    }

  private:
    static auto inner_create_stream() -> std::shared_ptr<stream_t>;
    static void set_device(int device);
    Backend() = default;
    int m_device = 0;
  };

  template <typename BackendT>
  class L2Flusher {
  public:
    static auto instance() -> L2Flusher<BackendT> * {
      static L2Flusher<BackendT> flusher;
      return &flusher;
    }
    void flush(std::shared_ptr<typename BackendT::stream_t> stream);
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
    void block(std::shared_ptr<typename BackendT::stream_t> stream, double timeout);
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
