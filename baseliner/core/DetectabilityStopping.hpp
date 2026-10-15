#ifndef BASELINER_CORE_DETECTABILITYSTOPPING_HPP
#define BASELINER_CORE_DETECTABILITYSTOPPING_HPP

#include <baseliner/core/Durations.hpp>
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <cstddef>
#include <vector>

namespace Baseliner {

  namespace Stats {

    /**
     * Lag-1 Autocorrelation Stat:
     * Estimates rho = C_1 / ((n - 1) * s^2)
     * where C_1 = sum_{i=0}^{n-2} (x_i - mean) * (x_{i+1} - mean)
     * Reuses StandardDeviation directly for the denominator.
     */
    class Lag1Autocorrelation : public IStat<Lag1Autocorrelation, float, ExecutionTimeVector, Mean, StandardDeviation> {
    public:
      [[nodiscard]] auto name() const -> std::string override {
        return "lag1_autocorrelation";
      }
      [[nodiscard]] auto unit() const -> std::string override {
        return "";
      }
      [[nodiscard]] auto granularity() const -> MetricGranularity override {
        return MetricGranularity::EVERY_BATCH;
      }
      void calculate(float &value_to_update, const std::vector<float_milliseconds> &times, const float &mean,
                     const float &stdev) override;
    };

    /**
     * MAR1 Effective Sample Size Stat:
     * Computes m_eff = ((1 - rho) / (1 + rho)) * n for AR(1) dependence.
     * When rho <= 0, no positive autocorrelation penalty is applied (m_eff = n).
     */
    class EffectiveSampleSizeMAR1 : public IStat<EffectiveSampleSizeMAR1, float, Repetitions, Lag1Autocorrelation> {
    public:
      [[nodiscard]] auto name() const -> std::string override {
        return "effective_sample_size_mar1";
      }
      [[nodiscard]] auto unit() const -> std::string override {
        return "";
      }
      [[nodiscard]] auto granularity() const -> MetricGranularity override {
        return MetricGranularity::EVERY_BATCH;
      }
      void calculate(float &value_to_update, const size_t &repetitions, const float &rho) override;
    };

  } // namespace Stats

  /**
   * Detectability-Guided Stopping Criterion (TT-MDE):
   * Terminates execution early once the effective sample size m_eff reaches
   * the required resolution m_req to detect differences between delta_0 and delta_1
   * with false-positive control alpha and target power 1 - beta:
   *   m_req = ceil( 2 * CV^2 * (z_{1-alpha} + z_{1-beta})^2 / (delta_1 - delta_0)^2 )
   *
   * Completely stateless: queries data from StatsEngine to make decisions.
   */
  class DetectabilityStoppingCriterion : public StoppingCriterion {
  public:
    explicit DetectabilityStoppingCriterion(size_t max_repetitions = DEFAULT_MAX_REPETITION)
        : StoppingCriterion(max_repetitions) {
    }

    void register_stats() override;

  protected:
    void register_options() override;
    auto criterion_satisfied() -> bool override;

  private:
    static auto normal_quantile(double p) -> double;
    static auto compute_z_score(double alpha, double beta) -> double;

    float m_delta_0 = 0.005f;  // 0.5% negligible regression threshold
    float m_delta_1 = 0.010f;  // 1.0% meaningful regression threshold
    float m_alpha = 0.05f;     // 5% false-positive rate
    float m_beta = 0.20f;      // 20% false-negative rate (80% target power)
    size_t m_min_samples = 10; // Minimum repetitions before evaluating detectability
  };

} // namespace Baseliner

#endif // BASELINER_CORE_DETECTABILITYSTOPPING_HPP
