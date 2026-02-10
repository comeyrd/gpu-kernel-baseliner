#ifndef STOPPING_CRITERION_HPP
#define STOPPING_CRITERION_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Stats.hpp>

#include <vector>
namespace Baseliner {

  constexpr size_t DEFAULT_MAX_REPETITION = 500;
  constexpr size_t DEFAULT_BATCH_SIZE = 1;
  class StoppingCriterion : public IOptionConsumer {
  public:
    virtual void addTime(float_milliseconds execution_time);
    auto satisfied() -> bool;
    void register_options() override;
    virtual auto get_metrics() -> std::vector<Metric>;
    auto executionTimes() -> std::vector<float_milliseconds>;
    virtual void reset();

    void set_max_repetitions(const size_t &val) {
      m_max_repetitions = val;
    };
    auto get_max_repetitions() const -> size_t {
      return m_max_repetitions;
    }
    void set_m_batch_size(const size_t &val) {
      m_batch_size = val;
    }
    auto get_batch_size() const -> size_t {
      return m_batch_size;
    }
    auto get_execution_time_vector() -> std::vector<float_milliseconds> & {
      return m_execution_times_vector;
    }

  protected:
    virtual auto criterion_satisfied() -> bool;

  private:
    size_t m_max_repetitions = DEFAULT_MAX_REPETITION;
    size_t m_batch_size = DEFAULT_BATCH_SIZE;
    std::vector<float_milliseconds> m_execution_times_vector;
  };
  constexpr float MEASURE_PRECISION = 0.0005F;
  constexpr float DEFAULT_RELATIVE_ERROR_TH = 0.001F;
  constexpr size_t MAX_REPETITION_CI = 2000;
  constexpr size_t BATCH_SIZE_CI = 50;
  class ConfidenceIntervalMedianSC : public StoppingCriterion {
  public:
    void register_options() override;
    void addTime(float_milliseconds execution_time) override;
    ConfidenceIntervalMedianSC() {
      set_max_repetitions(MAX_REPETITION_CI);
      set_m_batch_size(BATCH_SIZE_CI);
    };
    void reset() override;
    auto get_metrics() -> std::vector<Metric> override;

  private:
    std::vector<float_milliseconds> m_sorted_execution_times_vector;
    std::vector<float_milliseconds> m_sorted_without_outliers_time_vector;
    auto criterion_satisfied() -> bool override;
    bool m_remove_outliers = true;
    float m_precision =
        MEASURE_PRECISION * 2; // Cuda Event Precision = 0.0005ms, Hip Event 0.001ms, SYCL events 0.001ms also ~
    float m_relative_error_th = DEFAULT_RELATIVE_ERROR_TH;
    float m_ci_low = 0;
    float m_ci_high = 0;
    float m_ci_width = 0;
    float m_median_absolute_dev = 0;
    float m_Q1 = 0;
    float m_Q3 = 0;
    float m_relative_error = 0;
    float m_median = 0;
    Stats::ConfidenceInterval confidence_compute;
    // HIP Reference
    // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html    //

    // Cuda Reference
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6

    // SYCL Reference https://github.khronos.org/SYCL_Reference/iface/event.html#event-profiling-descriptors 64-bit
    // timestamp that represents the number of nanoseconds that have elapsed since some implementation-defined timebase.
  };

} // namespace Baseliner
#endif // STOPPING_CRITERION_HPP