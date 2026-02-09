#ifndef BASELINER_STATS_HPP
#define BASELINER_STATS_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <cmath>

namespace Baseliner {

  namespace Stats {
    // TAKES IN SORTED
    std::tuple<std::vector<float_milliseconds>, float, float>
    RemoveOutliers(std::vector<float_milliseconds> &vec_float);
    // TAKES IN SORTED
    std::pair<float, float> MedianAbsoluteDeviation(std::vector<float_milliseconds> &vec_float);

    class ConfidenceInterval : OptionConsumer {
    public:
      void register_options() override;
      std::pair<size_t, size_t> compute(size_t sample_size);
      double nCR(size_t sample_size, size_t prob_increment);

    private:
      std::pair<size_t, size_t> compute_small_sample_ranks(size_t sample_size);
      std::pair<size_t, size_t> compute_large_sample_ranks(size_t sample_size);

      double get_z_score();

      double m_probability = 0.5;
      float m_confidence = 0.95f;
      size_t m_large_sample_threshold = 30;
    };
  } // namespace Stats
} // namespace Baseliner
#endif // BASELINER_STATS_HPP