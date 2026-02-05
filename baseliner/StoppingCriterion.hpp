#ifndef STOPPING_CRITERION_HPP
#define STOPPING_CRITERION_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <chrono>
#include <vector>
namespace Baseliner {

  class IStoppingCriterion : public OptionConsumer {
  public:
    virtual void addTime(std::chrono::duration<float, std::milli> execution_time) {
      m_execution_times_vector.push_back(execution_time);
    };
    virtual bool satisfied() = 0;
    virtual std::vector<Metric> get_metrics() {
      std::vector<Metric> metrics;

      metrics.push_back(Metric("execution_times", "ms", m_execution_times_vector));
      return metrics;
    };
    std::vector<std::chrono::duration<float, std::milli>> executionTimes() {
      return m_execution_times_vector;
    };
    virtual void reset() {
      m_execution_times_vector.clear();
    }

  private:
    std::vector<std::chrono::duration<float, std::milli>> m_execution_times_vector;
  };

  class FixedRepetitionStoppingCriterion final : public IStoppingCriterion {
  public:
    int max_repetitions = 500;
    void register_options() override {
      add_option("FixedRepetition", "nb_repetition", "Numbers of repetitions", max_repetitions);
    };
    void addTime(std::chrono::duration<float, std::milli> execution_time) override {
      IStoppingCriterion::addTime(execution_time);
      m_repetitions_done++;
    };
    bool satisfied() override {
      return (m_repetitions_done >= max_repetitions);
    };
    FixedRepetitionStoppingCriterion() {};
    void reset() override {
      IStoppingCriterion::reset();
      m_repetitions_done = 0;
    };

  private:
    int m_repetitions_done = 0;
  };
} // namespace Baseliner
#endif // STOPPING_CRITERION_HPP