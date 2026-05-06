#ifndef BASELINER_PRIMBENCH_PRIMBENCHSTAT_HPP
#define BASELINER_PRIMBENCH_PRIMBENCHSTAT_HPP
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
namespace Baseliner::Stats {
  class CoefficientOfVariation : public IStat<CoefficientOfVariation, float, BatchTimeVector,BatchCount> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "coefficient_of_variation";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "%";
    }

    void calculate(float &value_to_update, const std::vector<float_milliseconds> &batch_times,const size_t &count) override {
      // Use the configurable window size
      if (count < 2) {
        value_to_update = 100.0f;
        return;
      }

      size_t start = (batch_times.size() > m_window_size) ? batch_times.size() - m_window_size : 0;
      size_t n = batch_times.size() - start;

      double sum = 0.0;
      for (size_t i = start; i < batch_times.size(); ++i) {
        sum += batch_times[i].count();
      }
      double mean = sum / n;

      double sq_sum = 0.0;
      for (size_t i = start; i < batch_times.size(); ++i) {
        sq_sum += std::pow(batch_times[i].count() - mean, 2);
      }

      double stdev = std::sqrt(sq_sum / (n - 1));

      value_to_update = (mean > 0) ? static_cast<float>((stdev / mean) * 100.0) : 100.0f;
    }

    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }

  protected:
    void register_options() override {
      add_option("CoefficientOfVariation", "window_size", "Number of recent batches used to calculate noise",
                 m_window_size);
    }

  private:
    size_t m_window_size = 10;
  };
} // namespace Baseliner::Stats
#endif // BASELINER_PRIMBENCH_PRIMBENCHSTAT_HPP