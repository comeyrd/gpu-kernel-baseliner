#include <baseliner/Stats.hpp>
#include <baseliner/StoppingCriterion.hpp>
namespace Baseliner {

  //----------
  // Stopping criterion
  //

  bool StoppingCriterion::satisfied() {
    size_t vec_size = m_execution_times_vector.size();
    if (vec_size > m_max_repetitions) {
      return true;
    } else if (((vec_size % m_batch_size) == 0) && vec_size > 1) {
      return criterion_satisfied();
    } else {
      return false;
    }
  }
  bool StoppingCriterion::criterion_satisfied() {
    return false;
  }
  void StoppingCriterion::addTime(float_milliseconds execution_time) {
    m_execution_times_vector.push_back(execution_time);
  };

  std::vector<Metric> StoppingCriterion::get_metrics() {
    std::vector<Metric> metrics;
    metrics.push_back(Metric("execution_times", "ms", m_execution_times_vector));
    return metrics;
  };

  std::vector<float_milliseconds> StoppingCriterion::executionTimes() {
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
  };

  void ConfidenceIntervalMedianSC::addTime(float_milliseconds execution_time) {
    StoppingCriterion::addTime(execution_time);
    auto it = std::lower_bound(m_sorted_execution_times_vector.begin(), m_sorted_execution_times_vector.end(),
                               execution_time);
    m_sorted_execution_times_vector.insert(it, execution_time);
  };

  bool ConfidenceIntervalMedianSC::criterion_satisfied() {
    std::pair<size_t, size_t> jk = confidence_compute.compute(m_sorted_execution_times_vector.size());
    m_ci_high = m_sorted_execution_times_vector[jk.second - 1].count();
    m_ci_low = m_sorted_execution_times_vector[jk.first - 1].count();
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
    } else if (m_relative_error <= 0.01) {
      return true;
    } else {
      return false;
    }
  };
  std::vector<Metric> ConfidenceIntervalMedianSC::get_metrics() {
    std::vector<Metric> metrics;
    Metric execution_time = Metric("execution_times", "ms", m_execution_times_vector);
    execution_time.add_stat(MetricStats("sorted", m_sorted_execution_times_vector));
    execution_time.add_stat(MetricStats("without_outliers", m_sorted_without_outliers_time_vector));
    execution_time.add_stat(MetricStats("median_m_ci_width", m_ci_width));
    execution_time.add_stat(MetricStats("median_m_ci_high", m_ci_high));
    execution_time.add_stat(MetricStats("median_absolute_dev", m_median_absolute_dev));
    execution_time.add_stat(MetricStats("median_m_ci_low", m_ci_low));
    execution_time.add_stat(MetricStats("Q1", m_Q1));
    execution_time.add_stat(MetricStats("Q3", m_Q3));
    metrics.push_back(execution_time);
    return metrics;
  };
  void ConfidenceIntervalMedianSC::reset() {
    StoppingCriterion::reset();
  };

} // namespace Baseliner