#ifndef BASELINER_PRIMBENCH_PRIMBENCH_STOPPING
#define BASELINER_PRIMBENCH_PRIMBENCH_STOPPING

#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/primbench/PrimBenchStat.hpp>
namespace Baseliner {
  class VariationStoppingCriterion : public StoppingCriterion {
  public:
    VariationStoppingCriterion()
        : StoppingCriterion() {
      // primbench defaults
      set_max_repetitions(2000);
      set_m_batch_size(1); // Evaluate every batch
    }

    void register_stats() override {
      StoppingCriterion::register_stats();
      get_stats_engine()->register_stat<Stats::BatchTimeVector>();
      get_stats_engine()->register_stat<Stats::CoefficientOfVariation>();
      get_stats_engine()->register_stat<Stats::ExecutionTimeVector>();
    }

  protected:
    void register_options() override {
      StoppingCriterion::register_options();
      add_option("VariationSC", "noise_tolerance", "Noise tolerance in percent", m_noise_tolerance);
      add_option("VariationSC", "min_duration_ms", "Minimum duration before stopping (ms)", m_min_duration_ms);
    }

  private:
    auto criterion_satisfied() -> bool override {
      auto engine = get_stats_engine();

      const auto &exec_times = engine->get_result<Stats::ExecutionTimeVector>();
      float total_ms = 0;
      for (auto t : exec_times)
        total_ms += t.count();

      if (total_ms < m_min_duration_ms)
        return false;

      float current_cv = engine->get_result<Stats::CoefficientOfVariation>();

      return current_cv <= m_noise_tolerance;
    }

    float m_noise_tolerance = 1.0f;    // 1% tolerance
    float m_min_duration_ms = 1000.0f; // 1 second min
  };
} // namespace Baseliner
#endif // BASELINER_PRIMBENCH_PRIMBENCH_STOPPING