#ifndef ITIMER_HPP
#define ITIMER_HPP
template <typename stream_t>
class ITimer {
public:
  virtual void start(stream_t stream) = 0;
  virtual void stop(stream_t stream) = 0;
  virtual float time_elapsed() = 0;
};

#endif // ITIMER_HPP