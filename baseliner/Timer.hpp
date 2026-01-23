#ifndef ITIMER_HPP
#define ITIMER_HPP
#include <baseliner/Durations.hpp>
#include <chrono>
namespace Baseliner {
  class ITimer {
  public:
    virtual ~ITimer() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual float_milliseconds time_elapsed() = 0;
  };
} // namespace Baseliner
#endif // ITIMER_HPP