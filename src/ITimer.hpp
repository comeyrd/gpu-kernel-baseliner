#ifndef ITIMER_HPP
#define ITIMER_HPP
#include <chrono>
#include "Durations.hpp"
template <typename stream_t>
class ITimer {
public:
  virtual void start(stream_t stream) = 0;
  virtual void stop(stream_t stream) = 0;
  virtual float_milliseconds time_elapsed() = 0;
};

#endif // ITIMER_HPP