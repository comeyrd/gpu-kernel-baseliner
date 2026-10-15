#include <algorithm>
#include <baseliner/core/DetectabilityStopping.hpp>
#include <baseliner/core/stats/StatsType.hpp>
#include <cmath>
#include <cstddef>
#include <limits>

namespace Baseliner {

  namespace Stats {

    void Lag1Autocorrelation::calculate(float &value_to_update, const std::vector<float_milliseconds> &times,
                                        const float &mean, const float &stdev) {
      if (times.size() < 2 || stdev <= 1e-9f) {
        value_to_update = 0.0f;
        return;
      }

      const size_t n = times.size();
      double c1 = 0.0;
      for (size_t i = 0; i < n - 1; ++i) {
        c1 += (static_cast<double>(times[i].count()) - static_cast<double>(mean)) *
              (static_cast<double>(times[i + 1].count()) - static_cast<double>(mean));
      }

      const double denominator = static_cast<double>(n - 1) * static_cast<double>(stdev) * static_cast<double>(stdev);
      if (denominator <= 1e-12) {
        value_to_update = 0.0f;
        return;
      }

      const auto raw_rho = static_cast<float>(c1 / denominator);
      value_to_update = std::clamp(raw_rho, -0.99f, 0.99f);
    }

    void EffectiveSampleSizeMAR1::calculate(float &value_to_update, const size_t &repetitions, const float &rho) {
      if (repetitions < 2) {
        value_to_update = static_cast<float>(repetitions);
        return;
      }

      if (rho <= 0.0f) {
        value_to_update = static_cast<float>(repetitions);
        return;
      }

      const float factor = (1.0f - rho) / (1.0f + rho);
      const float effective = factor * static_cast<float>(repetitions);
      value_to_update = std::max(1.0f, effective);
    }

  } // namespace Stats

  void DetectabilityStoppingCriterion::register_stats() {
    StoppingCriterion::register_stats();
    auto engine = get_stats_engine();
    engine->register_stat<Stats::ExecutionTimeVector>();
    engine->register_stat<Stats::Mean>();
    engine->register_stat<Stats::StandardDeviation>();
    engine->register_stat<Stats::Lag1Autocorrelation>();
    engine->register_stat<Stats::EffectiveSampleSizeMAR1>();
  }

  void DetectabilityStoppingCriterion::register_options() {
    StoppingCriterion::register_options();
    add_option("DetectabilitySC", "delta_0", "Lower threshold for negligible slowdowns (default 0.005 for 0.5%)",
               m_delta_0);
    add_option("DetectabilitySC", "delta_1", "Upper threshold for meaningful regression (default 0.010 for 1.0%)",
               m_delta_1);
    add_option("DetectabilitySC", "alpha", "Significance level / false positive rate (default 0.05)", m_alpha);
    add_option("DetectabilitySC", "beta", "Type II error rate (1 - power, default 0.20 for 80% power)", m_beta);
    add_option("DetectabilitySC", "min_samples", "Minimum sample count before stopping evaluation (default 10)",
               m_min_samples);
  }

  // Peter Acklam's rational approximation for the normal quantile function Phi^-1(p)
  auto DetectabilityStoppingCriterion::normal_quantile(double p) -> double {
    constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,  -2.759285104469687e+02,
                             1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01, -1.328068155288572e+01};
    constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                             3.754408661907416e+00};

    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    p = std::clamp(p, 1e-9, 1.0 - 1e-9);

    if (p < p_low) {
      const double q = std::sqrt(-2.0 * std::log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p <= p_high) {
      const double q = p - 0.5;
      const double r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
             (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  auto DetectabilityStoppingCriterion::compute_z_score(double alpha, double beta) -> double {
    const double z_alpha = normal_quantile(1.0 - alpha);
    const double z_beta = normal_quantile(1.0 - beta);
    return z_alpha + z_beta;
  }

  auto DetectabilityStoppingCriterion::criterion_satisfied() -> bool {
    auto engine = get_stats_engine();
    const size_t reps = engine->get_result<Stats::Repetitions>();
    if (reps < m_min_samples) {
      return false;
    }

    const float stdev = engine->get_result<Stats::StandardDeviation>();
    const float mean = engine->get_result<Stats::Mean>();
    if (mean <= 0.0f || stdev <= 0.0f) {
      return false;
    }

    const float cv = stdev / mean;
    const float m_eff = engine->get_result<Stats::EffectiveSampleSizeMAR1>();

    const double delta = static_cast<double>(m_delta_1 - m_delta_0);
    if (delta <= 0.0) {
      return false;
    }

    const double Z = compute_z_score(static_cast<double>(m_alpha), static_cast<double>(m_beta));
    const double d_cv = static_cast<double>(cv);

    // Required sample size: m_req = (2 * CV^2 * Z^2) / (delta_1 - delta_0)^2
    const double m_req = (2.0 * d_cv * d_cv * Z * Z) / (delta * delta);

    return static_cast<double>(m_eff) >= m_req;
  }

} // namespace Baseliner
