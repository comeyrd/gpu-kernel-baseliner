#ifndef CPU_TIMER_HPP
#define CPU_TIMER_HPP
#include "Timer.hpp"
#include <chrono>
namespace Baseliner {
  class CpuTimer : public ITimer {
  private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    TimePoint start_time;
    TimePoint end_time;

  public:
    void start() override {
      start_time = Clock::now();
    }
    void stop() override {
      end_time = Clock::now();
    }
    float_milliseconds time_elapsed() override {
      return std::chrono::duration_cast<float_milliseconds>(end_time - start_time);
    }
  };
} // namespace Baseliner
#endif // CPU_TIMER_HPP