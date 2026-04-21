#ifndef BASELINER_CORE_TIMER_HPP
#define BASELINER_CORE_TIMER_HPP
#include <baseliner/core/Durations.hpp>
#include <baseliner/core/Error.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

namespace Baseliner {

  template <typename BackendT>
  class ITimer {
  public:
    using Stream = std::shared_ptr<typename BackendT::stream_t>;
    using Kernel = std::function<void(Stream &, typename BackendT::launch_result_t &)>;
    using Funct = std::function<void(Stream &)>;

    virtual ~ITimer() = default;

    virtual void init(Stream &stream) = 0;
    virtual void measure_before(Stream &stream) = 0;
    virtual void measure_consume(typename BackendT::launch_result_t event);
    virtual void measure_after(Stream &stream) = 0;
    virtual auto elapsed() -> float_milliseconds = 0;

    virtual void init_batch(Stream &stream, size_t batch_size, bool is_blocking) = 0;
    virtual void measure_batch_before(Stream &stream) = 0;
    virtual void measure_batch_consume(typename BackendT::launch_result_t event) = 0;
    virtual void measure_batch_after(Stream &stream) = 0;
    virtual auto elapsed_batch() -> std::vector<float_milliseconds> = 0;
  };
  template <typename BackendT>
  class CpuTimer final : public ITimer<BackendT> {
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Stream = typename ITimer<BackendT>::Stream;
    using Kernel = typename ITimer<BackendT>::Kernel;
    using Funct = std::function<void(Stream &)>;

  public:
    void init(Stream &stream) override {
      if (m_state != State::Idle) {
        throw Errors::timer_init_on_not_idle();
      }
      reset();
      m_state = State::Single;
      m_starts.resize(1);
      m_stops.resize(1);
    }

    void measure_before(Stream &stream) override {
      if (m_state != State::Single) {
        throw Errors::timer_not_single_state("measure_before()");
      }
      m_starts[0] = Clock::now();
    }

    // No-op for CPU: GPU launch_result_t events are irrelevant here
    void measure_consume(typename BackendT::launch_result_t event) override {
    }

    void measure_after(Stream &stream) override {
      if (m_state != State::Single) {
        throw Errors::timer_not_single_state("measure_after()");
      }
      BackendT::instance()->synchronize(stream);
      m_stops[0] = Clock::now();
    }

    auto elapsed() -> float_milliseconds override {
      if (m_state != State::Single) {
        throw Errors::timer_not_single_state("elapsed()");
      }
      m_state = State::Idle;
      return float_milliseconds(m_stops[0] - m_starts[0]);
    }

    void init_batch(Stream &stream, size_t batch_size, bool is_blocking) override {
      if (m_state != State::Idle) {
        throw Errors::timer_init_on_not_idle();
      }
      reset();
      m_state = State::Batch;
      m_batch_size = batch_size;
      m_starts.reserve(batch_size);
      m_stops.reserve(batch_size);
      m_is_blocking = is_blocking;
      if (m_is_blocking) {
        std::cout << "[Baseliner][CpuTimer] Warning : using CpuTimer while using a blocking kernel gives back wrong "
                     "measurements";
      }
    }

    void measure_batch_before(Stream &stream) override {
      if (m_state != State::Batch) {
        throw Errors::timer_not_batch("measure_batch_before()");
      }
      m_starts.push_back(Clock::now());
    }

    // No-op for CPU
    void measure_batch_consume(typename BackendT::launch_result_t event) override {
    }

    void measure_batch_after(Stream &stream) override {
      if (m_state != State::Batch) {
        throw Errors::timer_not_batch("measure_batch_after()");
      }
      if (!m_is_blocking) {
        BackendT::instance()->synchronize(stream);
      }
      m_stops.push_back(Clock::now());
    }

    auto elapsed_batch() -> std::vector<float_milliseconds> override {
      if (m_state != State::Batch) {
        throw Errors::timer_not_batch("elapsed_batch()");
      }
      std::vector<float_milliseconds> results;
      results.reserve(m_starts.size());
      for (size_t i = 0; i < m_starts.size(); ++i) {
        results.emplace_back(m_stops[i] - m_starts[i]);
      }
      m_state = State::Idle;
      return results;
    }

  private:
    enum class State {
      Idle,
      Single,
      Batch
    };

    void reset() {
      m_starts.clear();
      m_stops.clear();
      m_batch_size = 0;
    }

    State m_state = State::Idle;
    bool m_is_blocking{false};
    size_t m_batch_size = 0;
    std::vector<TimePoint> m_starts;
    std::vector<TimePoint> m_stops;
  };
} // namespace Baseliner
#endif // BASELINER_CORE_TIMER_HPP