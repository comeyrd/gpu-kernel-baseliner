#ifndef BASELINER_NVBENCH_STRDELSTOPPING_HPP
#define BASELINER_NVBENCH_STRDELSTOPPING_HPP

#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <cmath>

namespace Baseliner {

  class StdRelStoppingCriterion : public StoppingCriterion {
  public:
    StdRelStoppingCriterion()
        : StoppingCriterion(200000) {
    }

    void register_stats() override {
      StoppingCriterion::register_stats();
      get_stats_engine()->register_stat<Stats::ExecutionTimeVector>();
      get_stats_engine()->register_stat<Stats::Mean>();
      get_stats_engine()->register_stat<Stats::RelativeStandardDeviation>();
      get_stats_engine()->register_stat<Stats::RelativeStandardDeviationVector>();
      get_stats_engine()->register_stat<Stats::GpuAccumulatedTime>();
    }

  protected:
    void register_options() override {
      StoppingCriterion::register_options();
      add_option("StdRelSC", "min_samples", "Minimum samples before checking noise", m_min_samples);
      add_option("StdRelSC", "min_time_ms", "Minimum accumulated GPU time (ms)", m_min_time_ms);
      add_option("StdRelSC", "max_noise", "Max relative stdev in percent", m_max_noise);
      add_option("StdRelSC", "noise_stability_threshold", "Rel stdev of noise below which noise is considered stable",
                 m_noise_stability_threshold);
      add_option("StdRelSC", "noise_stability_window", "Minimum noise entries before checking stability",
                 m_noise_stability_window);
    }

  private:
    auto criterion_satisfied() -> bool override {
      auto engine = get_stats_engine();

      if (engine->get_result<Stats::GpuAccumulatedTime>() < m_min_time_ms)
        return false;
      if (engine->get_result<Stats::RelativeStandardDeviation>() <= m_max_noise)
        return true;

      const auto &noise_vec = engine->get_result<Stats::RelativeStandardDeviationVector>();
      const size_t reps = engine->get_result<Stats::Repetitions>();

      if (noise_vec.size() > m_noise_stability_window) {
        float current = noise_vec.back();
        if (current <= 0.0f) {
          return false;
        }

        size_t start = (noise_vec.size() > m_ring_buffer_size) ? noise_vec.size() - m_ring_buffer_size : 0;

        double sq_sum = 0.0;
        size_t count = noise_vec.size() - start;
        for (size_t i = start; i < noise_vec.size(); ++i) {
          double diff = noise_vec[i] - current;
          sq_sum += diff * diff;
        }
        double stdev = std::sqrt(sq_sum / (count - 1));
        double rel_stdev = stdev / current;
        if (rel_stdev < m_noise_stability_threshold) {
          return true;
        }
      }

      return false;
    }

    size_t m_min_samples = 10;
    float m_min_time_ms = 500.0f;
    float m_max_noise = 0.5f;                  // 0.5%
    double m_noise_stability_threshold = 0.05; // 5% — noise of the noise
    size_t m_noise_stability_window = 64;
    size_t m_ring_buffer_size = 512;
  };

} // namespace Baseliner
#endif