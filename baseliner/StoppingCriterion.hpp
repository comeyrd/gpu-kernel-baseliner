#ifndef STOPPING_CRITERION_HPP
#define STOPPING_CRITERION_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/stats/Stats.hpp>
#include <baseliner/stats/StatsEngine.hpp>

namespace Baseliner {

  constexpr size_t DEFAULT_MAX_REPETITION = 500;
  constexpr size_t DEFAULT_BATCH_SIZE = 1;
  class StoppingCriterion : public IOption {
  public:
    auto satisfied() -> bool;
    void set_stats_engine(std::shared_ptr<Stats::StatsEngine> &engine) {
      m_stats_engine = engine;
      register_stats();
    }
    virtual void register_stats();
    void set_max_repetitions(const size_t &val) {
      m_max_repetitions = val;
    };
    [[nodiscard]] auto get_max_repetitions() const -> size_t {
      return m_max_repetitions;
    }
    void set_m_batch_size(const size_t &val) {
      m_batch_size = val;
    }
    [[nodiscard]] auto get_batch_size() const -> size_t {
      return m_batch_size;
    }
    StoppingCriterion(size_t max_repetition = DEFAULT_MAX_REPETITION, size_t batch_size = DEFAULT_BATCH_SIZE)
        : m_max_repetitions(max_repetition),
          m_batch_size(batch_size) {};

  protected:
    virtual auto criterion_satisfied() -> bool;
    void register_options() override;
    auto get_stats_engine() -> std::shared_ptr<Stats::StatsEngine> {
      return m_stats_engine;
    };

  private:
    std::shared_ptr<Stats::StatsEngine> m_stats_engine;
    size_t m_max_repetitions;
    size_t m_batch_size;
  };

  constexpr float MEASURE_PRECISION = 0.0005F;
  constexpr float DEFAULT_RELATIVE_ERROR_TH = 0.001F;
  constexpr size_t MAX_REPETITION_CI = 2000;
  constexpr size_t BATCH_SIZE_CI = 50;
  class ConfidenceIntervalMedianSC : public StoppingCriterion {
  public:
    ConfidenceIntervalMedianSC()
        : StoppingCriterion() {
      set_max_repetitions(MAX_REPETITION_CI);
      set_m_batch_size(BATCH_SIZE_CI);
    };
    void register_stats() override;

  protected:
    void register_options() override;

  private:
    auto criterion_satisfied() -> bool override;
    bool m_remove_outliers = true;
    float m_precision =
        MEASURE_PRECISION * 2; // Cuda Event Precision = 0.0005ms, Hip Event 0.001ms, SYCL events 0.001ms also ~
    float m_relative_error_th = DEFAULT_RELATIVE_ERROR_TH;
    // HIP Reference
    // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html    //

    // Cuda Reference
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6

    // SYCL Reference https://github.khronos.org/SYCL_Reference/iface/event.html#event-profiling-descriptors 64-bit
    // timestamp that represents the number of nanoseconds that have elapsed since some implementation-defined timebase.
  };

} // namespace Baseliner
#endif // STOPPING_CRITERION_HPP