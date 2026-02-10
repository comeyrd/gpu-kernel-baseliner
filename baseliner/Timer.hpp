#ifndef ITIMER_HPP
#define ITIMER_HPP
#include <baseliner/Durations.hpp>
#include <memory>
namespace Baseliner {
  class ITimer {
  public:
    virtual ~ITimer() = default;
    virtual void measure_start() = 0;
    virtual void measure_stop() = 0;
    virtual auto time_elapsed() -> float_milliseconds = 0;
  };
  template <typename stream_t>
  class IGpuTimer {
  public:
    virtual ~IGpuTimer() = default;
    IGpuTimer() = default;
    IGpuTimer(const IGpuTimer &) = delete;
    auto operator=(const IGpuTimer &) -> IGpuTimer & = delete;
    IGpuTimer(IGpuTimer &&) = delete;
    auto operator=(IGpuTimer &&) -> IGpuTimer & = delete;

  protected:
    virtual void measure_start(std::shared_ptr<stream_t> stream) = 0;
    virtual void measure_stop(std::shared_ptr<stream_t> stream) = 0;
    virtual auto time_elapsed() -> float_milliseconds = 0;
  };
} // namespace Baseliner
#endif // ITIMER_HPP