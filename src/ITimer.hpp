#ifndef ITIMER_HPP
#define ITIMER_HPP
#include "Durations.hpp"
#include <chrono>
class ITimer {
public:
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual float_milliseconds time_elapsed() = 0;
};

#endif // ITIMER_HPP