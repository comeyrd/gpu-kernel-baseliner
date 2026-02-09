#include <algorithm>
#include <baseliner/Stats.hpp>
#include <cmath>

namespace Baseliner {
  namespace Stats {
    std::vector<float_milliseconds> RemoveOutliers(std::vector<float_milliseconds> &vec_float) {
      size_t n = vec_float.size();
      size_t Q1_ix = static_cast<size_t>(std::floor(n / 4));
      float Q1 = vec_float[Q1_ix].count();
      size_t Q3_IX = static_cast<size_t>(std::floor((3 * n) / 4));
      float Q3 = vec_float[Q3_IX].count();

      float IQR = Q3 - Q1;

      float_milliseconds lower_fence = static_cast<float_milliseconds>(Q1 - (1.5f * IQR));
      float_milliseconds upper_fence = static_cast<float_milliseconds>(Q3 + (1.5f * IQR));

      auto it_start = std::lower_bound(vec_float.begin(), vec_float.end(), lower_fence);

      auto it_end = std::upper_bound(it_start, vec_float.end(), upper_fence);

      return std::vector<float_milliseconds>(it_start, it_end);
    };

    float MedianAbsoluteDeviation(std::vector<float_milliseconds> &vec_float) {
      if (vec_float.empty())
        return 0.0f;
      size_t n = vec_float.size();
      size_t middle = static_cast<size_t>(std::floor(n / 2));
      float median = vec_float[middle].count();

      std::vector<float> deviations;
      deviations.reserve(n);
      for (const auto &t : vec_float) {
        deviations.push_back(std::abs(t.count() - median));
      }

      std::sort(deviations.begin(), deviations.end());
      return deviations[middle];
    }
    // COMPUTED WITH tools/z-score-array-generator.py
    const static double MEDIAN_Z_SCORES[100] = {
        0.012533, 0.024942, 0.037355, 0.049773, 0.062199, 0.074635, 0.087082, 0.099543, 0.112019, 0.124513,
        0.137026, 0.149561, 0.162119, 0.174703, 0.187314, 0.199956, 0.212629, 0.225337, 0.238081, 0.250864,
        0.263688, 0.276556, 0.289469, 0.302431, 0.315444, 0.328511, 0.341634, 0.354816, 0.368060, 0.381369,
        0.394746, 0.408194, 0.421716, 0.435316, 0.448996, 0.462761, 0.476615, 0.490560, 0.504602, 0.518744,
        0.532990, 0.547345, 0.561814, 0.576402, 0.591113, 0.605954, 0.620929, 0.636044, 0.651307, 0.666722,
        0.682298, 0.698041, 0.713959, 0.730060, 0.746352, 0.762845, 0.779549, 0.796472, 0.813627, 0.831025,
        0.848678, 0.866599, 0.884803, 0.903306, 0.922123, 0.941272, 0.960772, 0.980645, 1.000913, 1.021601,
        1.042736, 1.064347, 1.086466, 1.109131, 1.132380, 1.156258, 1.180814, 1.206103, 1.232188, 1.259140,
        1.287039, 1.315977, 1.346061, 1.377415, 1.410186, 1.444545, 1.480699, 1.518899, 1.559454, 1.602750,
        1.649277, 1.699676, 1.754803, 1.815846, 1.884518, 1.963432, 2.056888, 2.172765, 2.328247, 2.575829};

    void ConfidenceInterval::register_options() {
      add_option("ConfidenceInterval", "quartile",
                 "The quartile to process the confidence interval in (Q1 = 0.25, Q2 = 0.5 ...)", m_probability);
      add_option("ConfidenceInterval", "confidence", "The aimed percentage confidence interval", m_confidence);
    };
    std::pair<size_t, size_t> ConfidenceInterval::compute(size_t sample_size) {
      if (sample_size == 0) {
        return {0, 0};
      } else if (sample_size < m_large_sample_threshold) {
        return compute_small_sample_ranks(sample_size);
      } else {
        return compute_large_sample_ranks(sample_size);
      }
    };
    double ConfidenceInterval::get_z_score() {
      float clamped_confidence = std::clamp(m_confidence, 0.0f, 1.0f);
      size_t index = static_cast<size_t>(std::floor(clamped_confidence * 100));
      if (index >= 0 && index < 100) {
        return MEDIAN_Z_SCORES[index];
      } else {
        throw std::runtime_error("Unreachable code");
      }
    }

    double ConfidenceInterval::nCR(size_t sample_size, size_t prob_increment) {
      if (prob_increment > sample_size)
        return 0;
      if (prob_increment == 0 || prob_increment == sample_size)
        return 1;
      if (prob_increment > sample_size / 2)
        prob_increment = sample_size - prob_increment;
      double res = 1;
      for (size_t i = 1; i <= prob_increment; ++i) {
        res = res * (static_cast<double>(sample_size - prob_increment + i)) / static_cast<double>(i);
      }
      return res;
    };

    std::pair<size_t, size_t> ConfidenceInterval::compute_small_sample_ranks(size_t sample_size) {
      double alpha_half = (1.0 - static_cast<double>(m_confidence)) / 2.0;
      size_t j = 1;

      // Find largest j such that the cumulative binomial probability
      // for p=0.5 is less than or equal to alpha/2.
      double cumulative_prob = 0.0;
      for (size_t i = 0; i < sample_size / 2; ++i) {
        // Binomial probability: P(X = i) for p=0.5 is nCr(n, i) * 0.5^n.
        double p_i = nCR(sample_size, i) * std::pow(0.5, static_cast<double>(sample_size));

        if (cumulative_prob + p_i <= alpha_half) {
          cumulative_prob += p_i;
          j = i + 1; // Rank is 1-based index.
        } else {
          break;
        }
      }

      // k is the symmetric upper rank.
      size_t k = sample_size - j + 1;
      return {j, k};
    }

    std::pair<size_t, size_t> ConfidenceInterval::compute_large_sample_ranks(size_t sample_size) {
      double eta = get_z_score();
      double d_sample = static_cast<double>(sample_size);
      double sqrt_part = eta * std::sqrt(d_sample * m_probability * (1 - m_probability));
      size_t j = static_cast<size_t>(std::floor(d_sample * m_probability - sqrt_part));
      size_t k = static_cast<size_t>(std::ceil(d_sample * m_probability + sqrt_part) + 1);
      return {j, k};
    }

  } // namespace Stats
} // namespace Baseliner