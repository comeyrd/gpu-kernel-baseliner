#ifndef BASELINER_STATS_HPP
#define BASELINER_STATS_HPP
#include <algorithm>
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsType.hpp>
#include <cmath>
#include <cstddef>
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
  };
  // For throughput calculations
  class ByteNumbers : public Imetric<ByteNumbers, size_t> {
  public:
    using Imetric<ByteNumbers, size_t>::Imetric; // Needs this for defaults
    [[nodiscard]] auto name() const -> std::string override {
      return "number_of_bytes_per_kernel";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
  };
  class SetupTime : public Imetric<SetupTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "setup_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
  };
  class TeardownTime : public Imetric<TeardownTime, float_milliseconds> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "teardown_time";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
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
  };
  class Repetitions : public IStat<Repetitions, size_t> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "Repetitions";
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }
    void calculate(Repetitions::type &value_to_update) override {
      value_to_update = value_to_update + 1;
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
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
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
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
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };

  class Median : public IStat<Median, float, SortedExecutionTimeVector> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "median";
    }
    void calculate(Median::type &value_to_update, const typename SortedExecutionTimeVector::type &inputs) override {
      const auto middle = static_cast<size_t>(std::floor(inputs.size() / 2));
      value_to_update = inputs[middle].count();
    };
    [[nodiscard]] auto unit() const -> std::string override {
      return "ms";
    }
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };
  };

  class MedianItemTroughput : public IStat<MedianItemTroughput, float, Median, ByteNumbers> {
    [[nodiscard]] auto name() const -> std::string override {
      return "MedianThroughput";
    }
    void calculate(MedianItemTroughput::type &value_to_update, const typename Median::type &median,
                   const typename ByteNumbers::type &nb_bytes) override {
      auto bytes = static_cast<double>(nb_bytes);
      auto seconds = static_cast<double>(median);

      if (seconds > 0) {
        value_to_update = static_cast<float>(bytes / (seconds * 1e9));
      } else {
        value_to_update = 0.0f;
      }
    };
    [[nodiscard]] auto unit() const -> std::string override {
      return "GB/s";
    }
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };
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
      const auto quarter = static_cast<size_t>(std::floor(inputs.size() / 4));
      value_to_update = inputs[quarter].count();
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };
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
      const auto three_quarter = static_cast<size_t>(std::floor(3 * inputs.size() / 4));
      value_to_update = inputs[three_quarter].count();
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };
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
      const ConfidenceInterval<size_t> bounds = get_confidence_interval(inputs.size());
      value_to_update = {inputs[bounds.high - 1], inputs[bounds.low - 1]};
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };

  protected:
    void register_options() override;

  private:
    auto get_confidence_interval(size_t sample_size) const -> ConfidenceInterval<size_t>;
    static auto nCR(size_t sample_size, size_t prob_increment) -> double;
    auto compute_small_sample_ranks(size_t sample_size) const -> ConfidenceInterval<size_t>;
    auto compute_large_sample_ranks(size_t sample_size) const -> ConfidenceInterval<size_t>;
    auto get_z_score() const -> double;

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

      const float InterQuartileRange = Q3_ - Q1_;

      auto lower_fence = static_cast<float_milliseconds>(Q1_ - (IQR_OUTLIER_RANGE * InterQuartileRange));
      auto upper_fence = static_cast<float_milliseconds>(Q3_ + (IQR_OUTLIER_RANGE * InterQuartileRange));

      auto it_start = std::lower_bound(sorted_vec.begin(), sorted_vec.end(), lower_fence);

      auto it_end = std::upper_bound(it_start, sorted_vec.end(), upper_fence);

      value_to_update = std::vector<float_milliseconds>(it_start, it_end);
    }
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };

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
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::ON_DEMAND;
    };
    void calculate(float &value_to_update, const typename SortedExecutionTimeVector::type &sorted_vec,
                   const typename Median::type &median) override {
      std::vector<float> deviations;
      deviations.reserve(sorted_vec.size());
      for (const auto &item : sorted_vec) {
        deviations.push_back(std::abs(item.count() - median));
      }
      std::sort(deviations.begin(), deviations.end());
      const auto middle = static_cast<size_t>(std::floor(sorted_vec.size() / 2));
      value_to_update = deviations[middle];
    }
  };

} // namespace Baseliner::Stats

#endif // BASELINER_STATS_HPP