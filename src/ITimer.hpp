#ifndef ITIMER_HPP
#define ITIMER_HPP
#include <chrono>

template <typename stream_t>
class ITimer {
public:
  virtual void start(stream_t stream) = 0;
  virtual void stop(stream_t stream) = 0;
  virtual std::chrono::duration<float, std::milli> time_elapsed() = 0;
};

#endif // ITIMER_HPP