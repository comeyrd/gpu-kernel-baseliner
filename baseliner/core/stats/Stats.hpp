#ifndef BASELINER_CORE_STATS_STATS_HPP
#define BASELINER_CORE_STATS_STATS_HPP
#include <algorithm>
#include <baseliner/core/Durations.hpp>

#include <baseliner/core/Options.hpp>

#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/StatsType.hpp>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>
namespace Baseliner::Stats {

  class ExecutionTime : public Imetric<ExecutionTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "execution_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };

  class ByteNumbers : public Imetric<ByteNumbers, size_t> {
  public:
    using Imetric<ByteNumbers, size_t>::Imetric;
    [[nodiscard]] auto name() const -> std::string override {
      return "memory_usage";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "bytes";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };

  class FLOPCount : public Imetric<FLOPCount, size_t> {
  public:
    using Imetric<FLOPCount, size_t>::Imetric;
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_usage";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "Flops";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };

  class BatchSize : public Imetric<BatchSize, int> {
  public:
    using Imetric<BatchSize, int>::Imetric;
    [[nodiscard]] auto name() const -> std::string override {
      return "batch_size";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }
  };
  class BatchTime : public Imetric<BatchTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "batch_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }
  };

  class ArithmeticIntensity : public IStat<ArithmeticIntensity, float, ByteNumbers, FLOPCount> {
    void calculate(ArithmeticIntensity::type &value_to_update, const typename ByteNumbers::type &byte,
                   const typename FLOPCount::type &flops) override {
      value_to_update = static_cast<float>(static_cast<double>(flops) / static_cast<double>(byte));
    }
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_intensity";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };
  class HostSetupTime : public Imetric<HostSetupTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "host_setup_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };
  class DeviceSetupTime : public Imetric<DeviceSetupTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "device_setup_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };

  class FetchResultsTime : public Imetric<FetchResultsTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "fetch_results_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };

  class WarmupTime : public Imetric<WarmupTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "warmup_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ONCE;
    }
  };

  class Repetitions : public IStat<Repetitions, size_t> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "repetitions";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    void calculate(Repetitions::type &value_to_update) override {
      value_to_update = value_to_update + 1;
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };
  class BatchCount : public IStat<BatchCount, size_t> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "batch_count";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    void calculate(BatchCount::type &value_to_update) override {
      value_to_update = value_to_update + 1;
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }
  };

  class BatchSizeVector : public IStat<BatchSizeVector, std::vector<int>, BatchSize> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "batch_size_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    void calculate(BatchSizeVector::type &value_to_update, const typename BatchSize::type &input) override {
      value_to_update.push_back(input);
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }
  };

  class ExecutionTimeVector : public IStat<ExecutionTimeVector, std::vector<float_milliseconds>, ExecutionTime> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "execution_time_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(ExecutionTimeVector::type &value_to_update, const typename ExecutionTime::type &inputs) override {
      value_to_update.push_back(inputs);
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };
  class BatchTimeVector : public IStat<BatchTimeVector, std::vector<float_milliseconds>, BatchTime> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "batch_time_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(BatchTimeVector::type &value_to_update, const typename BatchTime::type &inputs) override {
      value_to_update.push_back(inputs);
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_BATCH;
    }
  };

  class Mean : public IStat<Mean, float, ExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "mean";
    }
    void calculate(Mean::type &value_to_update, const typename ExecutionTimeVector::type &inputs) override {
      double total = 0;
      for (const auto &input : inputs) {
        total += input.count();
      }
      value_to_update = static_cast<float>(total / static_cast<double>(inputs.size()));
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };
  class HarmonicMean : public IStat<HarmonicMean, float, ExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "harmonic_mean";
    }
    void calculate(HarmonicMean::type &value_to_update, const typename ExecutionTimeVector::type &inputs) override {
      if (inputs.empty()) {
        value_to_update = 0.0f;
        return;
      }
      double sum_of_reciprocals = 0.0;
      size_t count = 0;

      for (const auto &input : inputs) {
        auto val = static_cast<double>(input.count());
        if (val > 0.0) {
          sum_of_reciprocals += 1.0 / val;
          count++;
        }
      }

      if (count > 0) {
        value_to_update = static_cast<float>(static_cast<double>(count) / sum_of_reciprocals);
      } else {
        value_to_update = 0.0f;
      }
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  class SortedExecutionTimeVector
      : public IStat<SortedExecutionTimeVector, std::vector<float_milliseconds>, ExecutionTime> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "sorted_execution_time_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(SortedExecutionTimeVector::type &value_to_update,
                   const typename ExecutionTime::type &inputs) override {
      auto iterator = std::lower_bound(value_to_update.begin(), value_to_update.end(), inputs);
      value_to_update.insert(iterator, inputs);
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };

  class Median : public IStat<Median, float, SortedExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "median";
    }
    void calculate(Median::type &value_to_update, const typename SortedExecutionTimeVector::type &inputs) override {
      if (!inputs.empty()) {
        const auto middle = static_cast<size_t>(std::floor(inputs.size() / 2));
        value_to_update = inputs[middle].count();
      } else {
        value_to_update = 0;
      }
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  class DataThroughput : public IStat<DataThroughput, float, HarmonicMean, ByteNumbers> {
    [[nodiscard]] auto name() const -> std::string override {
      return "memory_bandwidth";
    }
    void calculate(DataThroughput::type &value_to_update, const typename HarmonicMean::type &median,
                   const typename ByteNumbers::type &nb_bytes) override {
      auto bytes = static_cast<double>(nb_bytes);
      auto seconds = static_cast<double>(median);
      if (seconds > 0) {
        value_to_update = static_cast<float>(bytes / (seconds * 1e6));
      } else {
        value_to_update = 0.0F;
      }
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GB/s";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  class FLOPThroughput : public IStat<FLOPThroughput, float, HarmonicMean, FLOPCount> {
    [[nodiscard]] auto name() const -> std::string override {
      return "arithmetic_bandwidth";
    }
    void calculate(FLOPThroughput::type &value_to_update, const typename HarmonicMean::type &median,
                   const typename FLOPCount::type &nb_flops) override {
      auto flops = static_cast<double>(nb_flops);
      auto miliseconds = static_cast<double>(median);
      if (miliseconds > 0) {
        value_to_update = static_cast<float>(flops / (miliseconds * 1e6));
      } else {
        value_to_update = 0.0F;
      }
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "GFLOP/S";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  class Q1 : public IStat<Q1, float, SortedExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "Q1";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(Q1::type &value_to_update, const typename SortedExecutionTimeVector::type &inputs) override {
      if (!inputs.empty()) {
        const auto quarter = static_cast<size_t>(std::floor(inputs.size() / 4));
        value_to_update = inputs[quarter].count();
      } else {
        value_to_update = 0;
      }
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  class Q3 : public IStat<Q3, float, SortedExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "Q3";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(Q3::type &value_to_update, const typename SortedExecutionTimeVector::type &inputs) override {
      if (!inputs.empty()) {
        const auto three_quarter = static_cast<size_t>(std::floor(3 * inputs.size() / 4));
        value_to_update = inputs[three_quarter].count();
      } else {
        value_to_update = 0;
      }
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
  };

  constexpr double MEDIAN = 0.5F;
  constexpr double CONFIDENCE_95_PERCENT = 0.95F;
  constexpr size_t LARGE_SAMPLE_TH = 30;

  class MedianConfidenceInterval
      : public IStat<MedianConfidenceInterval, ConfidenceInterval<float_milliseconds>, SortedExecutionTimeVector> {
    [[nodiscard]] auto name() const -> std::string override {
      return "median_ci";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }

  public:
    void calculate(MedianConfidenceInterval::type &value_to_update,
                   const typename SortedExecutionTimeVector::type &inputs) override {
      if (!inputs.empty()) {
        const ConfidenceInterval<size_t> bounds = get_confidence_interval(inputs.size());
        value_to_update = {inputs[bounds.high - 1], inputs[bounds.low - 1]};
      } else {
        value_to_update = MedianConfidenceInterval::type{};
      }
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }

  protected:
    void register_options() override;

  private:
    [[nodiscard]] auto get_confidence_interval(size_t sample_size) const -> ConfidenceInterval<size_t>;
    static auto n_cr(size_t sample_size, size_t prob_increment) -> double;
    [[nodiscard]] auto compute_small_sample_ranks(size_t sample_size) const -> ConfidenceInterval<size_t>;
    [[nodiscard]] auto compute_large_sample_ranks(size_t sample_size) const -> ConfidenceInterval<size_t>;
    [[nodiscard]] auto get_z_score() const -> double;

    double m_probability = MEDIAN;
    float m_confidence = CONFIDENCE_95_PERCENT;
    size_t m_large_sample_threshold = LARGE_SAMPLE_TH;
  };

  constexpr float IQR_OUTLIER_RANGE = 1.5F;

  class WithoutOutliers
      : public IStat<WithoutOutliers, std::vector<float_milliseconds>, SortedExecutionTimeVector, Q1, Q3> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "without_outliers";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(std::vector<float_milliseconds> &value_to_update,
                   const typename SortedExecutionTimeVector::type &sorted_vec, const typename Q1::type &Q1_,
                   const typename Q3::type &Q3_) override {
      if (sorted_vec.size() > 0) {
        const float InterQuartileRange = Q3_ - Q1_;
        auto lower_fence = static_cast<float_milliseconds>(Q1_ - (IQR_OUTLIER_RANGE * InterQuartileRange));
        auto upper_fence = static_cast<float_milliseconds>(Q3_ + (IQR_OUTLIER_RANGE * InterQuartileRange));
        auto it_start = std::lower_bound(sorted_vec.begin(), sorted_vec.end(), lower_fence);
        auto it_end = std::upper_bound(it_start, sorted_vec.end(), upper_fence);
        value_to_update = std::vector<float_milliseconds>(it_start, it_end);
      }
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }

  protected:
    void register_options() override;

  private:
    float m_i_q_r_outlier_range = IQR_OUTLIER_RANGE;
  };

  class MedianAbsoluteDeviation : public IStat<MedianAbsoluteDeviation, float, SortedExecutionTimeVector, Median> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "MAD";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::ON_DEMAND;
    }
    void calculate(float &value_to_update, const typename SortedExecutionTimeVector::type &sorted_vec,
                   const typename Median::type &median) override {
      if (sorted_vec.size() > 0) {
        std::vector<float> deviations;
        deviations.reserve(sorted_vec.size());
        for (const auto &item : sorted_vec) {
          deviations.push_back(std::abs(item.count() - median));
        }
        std::sort(deviations.begin(), deviations.end());
        const auto middle = static_cast<size_t>(std::floor(sorted_vec.size() / 2));
        value_to_update = deviations[middle];
      }
    }
  };
  class RelativeStandardDeviation : public IStat<RelativeStandardDeviation, float, ExecutionTimeVector, Mean> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "relative_standard_deviation";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "%";
    }
    void calculate(float &value_to_update, const std::vector<float_milliseconds> &times, const float &mean) override {
      if (times.size() < 2 || mean <= 0.0f) {
        value_to_update = 100.0f;
        return;
      }
      double sq_sum = 0.0;
      for (const auto &t : times) {
        double diff = t.count() - mean;
        sq_sum += diff * diff;
      }
      double stdev = std::sqrt(sq_sum / (times.size() - 1));
      value_to_update = static_cast<float>((stdev / mean) * 100.0);
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };
  class RelativeStandardDeviationVector
      : public IStat<RelativeStandardDeviationVector, std::vector<float>, RelativeStandardDeviation> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "relative_standard_deviation_vector";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "%";
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
    void calculate(std::vector<float> &value_to_update, const float &noise) override {
      value_to_update.push_back(noise);
    }
  };

  class GpuAccumulatedTime : public IStat<GpuAccumulatedTime, float, ExecutionTime> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "gpu_accumulated_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    void calculate(float &value_to_update, const float_milliseconds &exec_time) override {

      value_to_update += exec_time.count();
    }
    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
  };

} // namespace Baseliner::Stats

#endif // BASELINER_STATS_HPP