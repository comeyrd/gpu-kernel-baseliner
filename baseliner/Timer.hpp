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

} // namespace Baseliner
#endif // ITIMER_HPP