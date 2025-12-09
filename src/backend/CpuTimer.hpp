#ifndef CPU_TIMER_HPP
#define CPU_TIMER_HPP
#include "Backend.hpp"
#include "Timer.hpp"
#include <chrono>
namespace Baseliner {
  namespace Backend {
    template <typename D>
    class CpuTimer : public D::GpuTimer {
    private:
      using Clock = std::chrono::high_resolution_clock;
      using TimePoint = Clock::time_point;

      TimePoint start_time;
      TimePoint end_time;

    public:
      void start() override {
        start_time = Clock::now();
      };
      void stop() override {
        end_time = Clock::now();
      };
      float_milliseconds time_elapsed() override {
        return std::chrono::duration_cast<float_milliseconds>(end_time - start_time);
      };
    };
  } // namespace Backend
} // namespace Baseliner
#endif // CPU_TIMER_HPP