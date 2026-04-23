#ifndef BASELINER_NVBENCH_NVBENCHSTOPPING_HPP
#define BASELINER_NVBENCH_NVBENCHSTOPPING_HPP
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/nvbench/NvBenchStat.hpp>
namespace Baseliner {

  class EntropyStoppingCriterion : public StoppingCriterion {
  public:
    EntropyStoppingCriterion()
        : StoppingCriterion() {
      set_max_repetitions(2000);
      set_m_batch_size(1);
    }

    void register_stats() override {
      StoppingCriterion::register_stats();
      auto engine = get_stats_engine();

      engine->register_stat<Stats::BinningConfig>();
      engine->register_stat<Stats::FrequencyTracker>();
      engine->register_stat<Stats::EntropySlidingWindow>();
      engine->register_stat<Stats::EntropyRegression>();
    }

  protected:
    void register_options() override {
      StoppingCriterion::register_options();
      add_option("EntropySC", "max_angle", "Maximum allowed entropy slope angle (deg)", m_max_angle);
      add_option("EntropySC", "min_r2", "Minimum R-squared for regression stability", m_min_r2);
      add_option("EntropySC", "min_samples", "Minimum samples before checking entropy", m_min_samples);
    }

  private:
    auto criterion_satisfied() -> bool override {
      auto engine = get_stats_engine();

      const auto &reg = engine->get_result<Stats::EntropyRegression>();
      const auto &window = engine->get_result<Stats::EntropySlidingWindow>();

      // 1. Guard: Ensure we have enough data in the sliding window
      if (window.current_size < m_min_samples)
        return false;

      if (engine->get_result<Stats::Repetitions>() % 2 != 0)
        return false;

      if (!reg.is_valid)
        return false;

      const bool is_flat = reg.slope_deg <= m_max_angle;
      const bool is_stable = reg.r_squared >= m_min_r2;

      return is_flat && is_stable;
    }

    float m_max_angle = 0.048f;
    float m_min_r2 = 0.36f;
    size_t m_min_samples = 10;
  };

} // namespace Baseliner

#endif