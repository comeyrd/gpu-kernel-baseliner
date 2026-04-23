/*
 *  Copyright 2023 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cmath>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Baseliner::Utils {

  /**
   * Computes the arithmetic mean of a range.
   */
  template <typename Iter>
  auto compute_mean(Iter first, Iter last) -> double {
    const auto n = std::distance(first, last);
    if (n < 1)
      return 0.0;
    return std::accumulate(first, last, 0.0) / static_cast<double>(n);
  }

  /**
   * @brief Computes linear regression slope and intercept.
   * Assumes X starts at 0 and increments by 1 for every element.
   */
  template <typename Iter>
  auto compute_linear_regression(Iter first, Iter last) -> std::pair<double, double> {
    const auto n = static_cast<size_t>(std::distance(first, last));

    if (n < 2) {
      return {0.0, 0.0};
    }

    const double mean_y = compute_mean(first, last);
    // Arithmetic progression mean for x = [0, 1, 2, ... n-1]
    const double mean_x = (static_cast<double>(n) - 1.0) / 2.0;

    double numerator = 0.0;
    double denominator = 0.0;

    for (size_t i = 0; i < n; ++i, ++first) {
      const double x_diff = static_cast<double>(i) - mean_x;
      numerator += x_diff * (*first - mean_y);
      denominator += x_diff * x_diff;
    }

    const double slope = numerator / denominator;
    const double intercept = mean_y - slope * mean_x;

    return {slope, intercept};
  }

  inline auto rad2deg(double rad) -> double {
    return rad * 180.0 / M_PI;
  }

  inline auto slope2deg(double slope) -> double {
    return rad2deg(std::atan2(slope, 1.0));
  }

  class online_linear_regression {
  private:
    double m_sum_x = 0.0;
    double m_sum_y = 0.0;
    double m_sum_xy = 0.0;
    double m_sum_x2 = 0.0;
    double m_sum_y2 = 0.0;
    int64_t m_count = 0;

  public:
    online_linear_regression() = default;

    void update(std::pair<double, double> incoming) {
      const auto [x, y] = incoming;
      m_sum_x += x;
      m_sum_y += y;
      m_sum_xy += x * y;
      m_sum_x2 += x * x;
      m_sum_y2 += y * y;
      m_count++;
    }

    void slide_window(double y_out, double y_in) {
      m_sum_y -= y_out;
      m_sum_y += y_in;

      m_sum_y2 -= y_out * y_out;
      m_sum_y2 += y_in * y_in;
      m_sum_xy -= (m_sum_y - y_in);
      m_sum_xy += (static_cast<double>(m_count) - 1.0) * y_in;
    }

    [[nodiscard]] double slope() const {
      if (m_count < 2)
        return std::numeric_limits<double>::quiet_NaN();

      const double n = static_cast<double>(m_count);
      const double mean_x = m_sum_x / n;
      const double mean_y = m_sum_y / n;

      const double numerator = (m_sum_xy / n) - (mean_x * mean_y);
      const double denominator = (m_sum_x2 / n) - (mean_x * mean_x);

      if (std::abs(denominator) < 1e-12)
        return std::numeric_limits<double>::quiet_NaN();

      return numerator / denominator;
    }

    [[nodiscard]] double intercept() const {
      if (m_count < 2)
        return std::numeric_limits<double>::quiet_NaN();
      const double s = slope();
      if (!std::isfinite(s))
        return std::numeric_limits<double>::quiet_NaN();

      return (m_sum_y / static_cast<double>(m_count)) - s * (m_sum_x / static_cast<double>(m_count));
    }

    [[nodiscard]] double r_squared() const {
      if (m_count < 2)
        return 0.0;

      const double n = static_cast<double>(m_count);
      const double mean_y = m_sum_y / n;
      const double ss_tot = (m_sum_y2 / n) - (mean_y * mean_y);

      if (ss_tot < std::numeric_limits<double>::epsilon())
        return 1.0;

      const double s = slope();
      const double intercept_v = intercept();
      if (!std::isfinite(s) || !std::isfinite(intercept_v))
        return 0.0;

      const double mean_xy = m_sum_xy / n;
      const double mean_xx = m_sum_x2 / n;
      const double mean_x = m_sum_x / n;

      const double ss_res_scaled = s * ((mean_xy - s * mean_xx) + (mean_xy - intercept_v * mean_x)) +
                                   intercept_v * (mean_y - s * mean_x - intercept_v) + mean_y * (intercept_v - mean_y);

      return std::min(std::max(ss_res_scaled / ss_tot, 0.0), 1.0);
    }
  };

} // namespace Baseliner::Utils