
#include <baseliner/Durations.hpp>
#include <baseliner/Stats.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <cstddef>
namespace Baseliner {

  //----------
  // Stopping criterion
  //

  auto StoppingCriterion::satisfied() -> bool {
    if (!m_started) {
      m_started = true;
      return false;
    }
    const size_t &vec_size = m_stats_engine->get_result<Stats::Repetitions>();
    if (vec_size > m_max_repetitions) {
      return true;
    }
    if (((vec_size % m_batch_size) == 0) && vec_size > 1) {
      return criterion_satisfied();
    }
    return false;
  }
  auto StoppingCriterion::criterion_satisfied() -> bool {
    return false;
  }
  void StoppingCriterion::register_options() {
    add_option("StoppingCriterion", "batch_size", "Numbers of rerun to wait before reevaluating stopping criterion",
               m_batch_size);
    add_option("StoppingCriterion", "max_nb_repetition", "Maximum number of repetitions", m_max_repetitions);
  };

  //----------
  // ConfidenceIntervalMedianSC
  //

  void ConfidenceIntervalMedianSC::register_options() {
    StoppingCriterion::register_options();
    add_option("ConfidenceIntervalSC", "precision", "The aimed precision (ms)", m_precision);
    add_option("ConfidenceIntervalSC", "remove_outliers",
               "If true, it will remove outliers in accordance to the outliers parameters", m_remove_outliers);
    add_option("ConfidenceIntervalSC", "relative_error_th", "The threshold at which it will consider converged",
               m_relative_error_th);
  };

  auto ConfidenceIntervalMedianSC::criterion_satisfied() -> bool {
    auto engine = get_stats_engine();
    const ConfidenceInterval<float_milliseconds> &confidence_interval =
        engine->get_result<Stats::MedianConfidenceInterval>();
    const float_milliseconds m_ci_width = confidence_interval.high - confidence_interval.low;
    return m_ci_width.count() <= m_precision;
  };

} // namespace Baseliner