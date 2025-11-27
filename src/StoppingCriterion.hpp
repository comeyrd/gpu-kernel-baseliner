#ifndef STOPPING_CRITERION_HPP
#define STOPPING_CRITERION_HPP
#include "Durations.hpp"
#include "Options.hpp"
#include <chrono>
#include <vector>
namespace Baseliner {

  class IStoppingCriterion {
  public:
    virtual void addTime(std::chrono::duration<float, std::milli> execution_time) {
      m_execution_times_vector.push_back(execution_time);
    };
    virtual bool satisfied() = 0;
    std::vector<std::chrono::duration<float, std::milli>> executionTimes() {
      return m_execution_times_vector;
    };
    void reset() {
      m_execution_times_vector.clear();
    }

  private:
    std::vector<std::chrono::duration<float, std::milli>> m_execution_times_vector;
  };

  class FixedRepetitionStoppingCriterion final : public IStoppingCriterion {
  public:
    void addTime(std::chrono::duration<float, std::milli> execution_time) override {
      IStoppingCriterion::addTime(execution_time);
      m_repetitions_done++;
    };
    bool satisfied() override {
      return (m_repetitions_done >= max_repetitions);
    };
    FixedRepetitionStoppingCriterion() {};

  private:
    int m_repetitions_done = 0;
    int max_repetitions = 500;
  };
} // namespace Baseliner
#endif // STOPPING_CRITERION_HPP