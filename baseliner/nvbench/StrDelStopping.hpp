#ifndef BASELINER_NVBENCH_STRDELSTOPPING_HPP
#define BASELINER_NVBENCH_STRDELSTOPPING_HPP
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/core/stats/Stats.hpp>
namespace Baseliner {
  class StdRelStoppingCriterion : public StoppingCriterion {
  public:
    StdRelStoppingCriterion()
        : StoppingCriterion(100000) {
    }

    void register_stats() override {
      StoppingCriterion::register_stats();
      get_stats_engine()->register_stat<Stats::ExecutionTimeVector>();
      get_stats_engine()->register_stat<Stats::Mean>();
      get_stats_engine()->register_stat<Stats::RelativeStandardDeviation>();
      get_stats_engine()->register_stat<Stats::GpuAccumulatedTime>();
    }

  protected:
    void register_options() override {
      StoppingCriterion::register_options();
      add_option("StdRelSC", "min_samples", "Minimum samples before checking noise", m_min_samples);
      add_option("StdRelSC", "min_time_ms", "Minimum accumulated GPU time (ms)", m_min_time_ms);
      add_option("StdRelSC", "max_noise", "Max relative stdev in percent", m_max_noise);
    }

  private:
    auto criterion_satisfied() -> bool override {
      auto engine = get_stats_engine();

      if (engine->get_result<Stats::Repetitions>() < m_min_samples)
        return false;

      if (engine->get_result<Stats::GpuAccumulatedTime>() < m_min_time_ms)
        return false;

      return engine->get_result<Stats::RelativeStandardDeviation>() <= m_max_noise;
    }

    size_t m_min_samples = 10;
    float m_min_time_ms = 500.0f;
    float m_max_noise = 0.5f;
  };
} // namespace Baseliner
#endif