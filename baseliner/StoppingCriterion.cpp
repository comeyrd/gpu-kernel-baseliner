#include <algorithm>
#include <baseliner/Durations.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Stats.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <cstddef>
#include <utility>
#include <vector>
namespace Baseliner {

  //----------
  // Stopping criterion
  //

  auto StoppingCriterion::satisfied() -> bool {
    const size_t vec_size = m_execution_times_vector.size();
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
  void StoppingCriterion::addTime(float_milliseconds execution_time) {
    m_execution_times_vector.push_back(execution_time);
  };

  auto StoppingCriterion::get_metrics() -> std::vector<Metric> {
    std::vector<Metric> metrics;
    metrics.push_back({"execution_times", "ms", m_execution_times_vector, {}});
    return metrics;
  };

  auto StoppingCriterion::executionTimes() -> std::vector<float_milliseconds> {
    return m_execution_times_vector;
  };

  void StoppingCriterion::reset() {
    m_execution_times_vector.clear();
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

  void ConfidenceIntervalMedianSC::addTime(float_milliseconds execution_time) {
    StoppingCriterion::addTime(execution_time);
    auto iterator = std::lower_bound(m_sorted_execution_times_vector.begin(), m_sorted_execution_times_vector.end(),
                                     execution_time);
    m_sorted_execution_times_vector.insert(iterator, execution_time);
  };

  auto ConfidenceIntervalMedianSC::criterion_satisfied() -> bool {
    const std::pair<size_t, size_t> bounds = confidence_compute.compute(m_sorted_execution_times_vector.size());
    m_ci_high = m_sorted_execution_times_vector[bounds.second - 1].count();
    m_ci_low = m_sorted_execution_times_vector[bounds.first - 1].count();
    m_ci_width = m_ci_high - m_ci_low;
    auto pair = Stats::MedianAbsoluteDeviation(m_sorted_execution_times_vector);
    m_median_absolute_dev = pair.first;
    m_median = pair.second;
    auto tuple = Stats::RemoveOutliers(m_sorted_execution_times_vector);
    m_sorted_without_outliers_time_vector = std::get<0>(tuple);
    m_Q1 = std::get<1>(tuple);
    m_Q3 = std::get<2>(tuple);
    m_relative_error = m_ci_width / m_median;
    if (m_ci_width <= m_precision && m_median_absolute_dev <= m_precision) {
      return true;
    }
    if (m_relative_error <= m_relative_error_th) {
      return true;
    }
    return false;
  };
  auto ConfidenceIntervalMedianSC::get_metrics() -> std::vector<Metric> {
    std::vector<Metric> metrics;
    Metric execution_time = {"execution_times", "ms", get_execution_time_vector(), {}};
    execution_time.m_v_stats.push_back({"sorted", m_sorted_execution_times_vector});
    execution_time.m_v_stats.push_back({"without_outliers", m_sorted_without_outliers_time_vector});
    execution_time.m_v_stats.push_back({"median_m_ci_width", m_ci_width});
    execution_time.m_v_stats.push_back({"median_m_ci_high", m_ci_high});
    execution_time.m_v_stats.push_back({"median_absolute_dev", m_median_absolute_dev});
    execution_time.m_v_stats.push_back({"relative_error", m_relative_error});
    execution_time.m_v_stats.push_back({"median_m_ci_low", m_ci_low});
    execution_time.m_v_stats.push_back({"Q1", m_Q1});
    execution_time.m_v_stats.push_back({"Q3", m_Q3});
    metrics.push_back(execution_time);
    return metrics;
  };
  void ConfidenceIntervalMedianSC::reset() {
    StoppingCriterion::reset();
  };

} // namespace Baseliner