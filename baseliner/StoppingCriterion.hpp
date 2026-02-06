#ifndef STOPPING_CRITERION_HPP
#define STOPPING_CRITERION_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Stats.hpp>

#include <vector>
namespace Baseliner {

  class StoppingCriterion : public OptionConsumer {
  public:
    virtual void addTime(float_milliseconds execution_time);
    bool satisfied();
    void register_options() override;
    virtual std::vector<Metric> get_metrics();
    std::vector<float_milliseconds> executionTimes();
    virtual void reset();

    size_t m_max_repetitions = 500;
    size_t m_batch_size = 1;

  protected:
    virtual bool criterion_satisfied();
    std::vector<float_milliseconds> m_execution_times_vector;
  };

  class ConfidenceIntervalMedianSC : public StoppingCriterion {
  public:
    void register_options() override;
    void addTime(float_milliseconds execution_time) override;
    ConfidenceIntervalMedianSC() {
      m_max_repetitions = 2000;
      m_batch_size = 50;
    };
    void reset() override;
    std::vector<Metric> get_metrics() override;

  private:
    std::vector<float_milliseconds> m_sorted_execution_times_vector;
    bool criterion_satisfied() override;
    bool m_remove_outliers = true;
    float m_precision = 0.0005f * 2; // Cuda Event Precision = 0.0005ms, Hip Event 0.001ms, SYCL events 0.001ms also ~
    float m_ci_low = 0;
    float m_ci_high = 0;
    float m_ci_width = 0;
    float m_median_absolute_dev = 0;
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