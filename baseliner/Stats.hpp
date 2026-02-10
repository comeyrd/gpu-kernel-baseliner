#ifndef BASELINER_STATS_HPP
#define BASELINER_STATS_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
namespace Baseliner::Stats {
  constexpr double MEDIAN = 0.5F;
  constexpr double CONFIDENCE_95_PERCENT = 0.95F;
  constexpr size_t LARGE_SAMPLE_TH = 30;
  // TAKES IN SORTED
  auto RemoveOutliers(std::vector<float_milliseconds> &vec_float)
      -> std::tuple<std::vector<float_milliseconds>, float, float>;
  // TAKES IN SORTED
  auto MedianAbsoluteDeviation(std::vector<float_milliseconds> &vec_float) -> std::pair<float, float>;

  class ConfidenceInterval : IOptionConsumer {
  public:
    void register_options() override;
    auto compute(size_t sample_size) -> std::pair<size_t, size_t>;
    static auto nCR(size_t sample_size, size_t prob_increment) -> double;

  private:
    auto compute_small_sample_ranks(size_t sample_size) const -> std::pair<size_t, size_t>;
    auto compute_large_sample_ranks(size_t sample_size) const -> std::pair<size_t, size_t>;

    auto get_z_score() const -> double;

    double m_probability = MEDIAN;
    float m_confidence = CONFIDENCE_95_PERCENT;
    size_t m_large_sample_threshold = LARGE_SAMPLE_TH;
  };
} // namespace Baseliner::Stats

#endif // BASELINER_STATS_HPP