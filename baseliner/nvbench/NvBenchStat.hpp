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

#ifndef BASELINER_CORE_STATS_NVBENCHSTAT_HPP
#define BASELINER_CORE_STATS_NVBENCHSTAT_HPP
#include <algorithm>
#include <baseliner/core/Durations.hpp>
#include <baseliner/core/Options.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <baseliner/core/stats/StatsType.hpp>
#include <baseliner/nvbench/NvBenchUtils.hpp>
#include <cmath>
#include <cstddef>
#include <deque>
#include <numeric>
#include <string>
#include <vector>
namespace Baseliner::Stats {

  /**
   * Shared parameters for NVBench-style measurement binning.
   */
  struct BinningParams {
    bool bin_keys = false;
    double resolution_us = 0.5;
    operator std::monostate() const {
      return std::monostate{};
    }
    // Helper to compute epsilon once (translates microseconds to milliseconds)
    [[nodiscard]] auto get_epsilon_ms() const -> double {
      return (resolution_us / 1000.0) * 2.0;
    }
  };

  /**
   *  A Stat that acts as a dependency provider for global benchmark options.
   * This executes ONCE at the start of the run.
   */
  class BinningConfig : public IStat<BinningConfig, BinningParams> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "binning_config";
    }

    void calculate(BinningConfig::type &params) override {
      params = m_params;
    }

    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }

  protected:
    void register_options() override {
      add_option("NVBench", "bin_keys", "Enable NVBench-style key binning", m_params.bin_keys);
      add_option("NVBench", "resolution_us", "Timing resolution in microseconds", m_params.resolution_us);
    }

  private:
    BinningParams m_params;
  };

  using FrequencyVector = std::vector<std::pair<float, size_t>>;

  /**
   * The data bundle passed to the next Stat in the chain.
   */
  struct TrackerResult {
    FrequencyVector data;
    size_t last_index;
    operator std::monostate() const {
      return std::monostate{};
    }
  };

  class FrequencyTracker : public IStat<FrequencyTracker, TrackerResult, ExecutionTime, BinningConfig> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "frequency_tracker";
    }

    void calculate(TrackerResult &result, const typename ExecutionTime::type &latest_time,
                   const BinningConfig::type &config) override {

      float key = latest_time.count();
      if (config.bin_keys) {
        const double epsilon = config.get_epsilon_ms();
        key = static_cast<float>(std::round(static_cast<double>(key) / epsilon) * epsilon);
      }

      auto it = std::lower_bound(result.data.begin(), result.data.end(), std::make_pair(key, size_t{0}),
                                 [](const auto &a, const auto &b) { return a.first < b.first; });

      size_t index = 0;

      if (it != result.data.end() && it->first == key) {
        it->second += 1;
        index = std::distance(result.data.begin(), it);
      } else {

        auto new_it = result.data.insert(it, {key, 1});
        index = std::distance(result.data.begin(), new_it);
      }
      result.last_index = index;
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }

    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
  };

  //     The signal emitted to the Regression Stat.
  // Tells the consumer exactly how the window shifted so it can update sums incrementally.

  struct EntropyWindowUpdate {
    float added_value;
    float evicted_value;
    bool was_sliding;
    size_t current_size;
    operator std::monostate() const {
      return std::monostate{};
    }
  };

  class EntropySlidingWindow : public IStat<EntropySlidingWindow, EntropyWindowUpdate, FrequencyTracker> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "entropy_window";
    }

    void calculate(EntropyWindowUpdate &update, const TrackerResult &tracker) override {
      m_total_samples++;

      const size_t new_count = tracker.data[tracker.last_index].second;
      const auto old_count = static_cast<double>(new_count - 1);
      const auto new_c = static_cast<double>(new_count);

      if (old_count > 0) {
        m_sum_count_log_counter += new_c * std::log2(new_c / old_count) + std::log2(old_count);
      } else {
        m_sum_count_log_counter += new_c * std::log2(new_c);
      }

      const auto n = static_cast<double>(m_total_samples);
      const float current_entropy = static_cast<float>(std::max(0.0, std::log2(n) - (m_sum_count_log_counter / n)));

      update.added_value = current_entropy;
      update.was_sliding = (m_internal_window.size() >= m_max_window_size);

      if (update.was_sliding) {
        update.evicted_value = m_internal_window.front();
        m_internal_window.pop_front(); // Use deque for efficient front removal
      } else {
        update.evicted_value = 0.0f;
      }

      m_internal_window.push_back(current_entropy);
      update.current_size = m_internal_window.size();
    }

    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }

  protected:
    void register_options() override {
      add_option("NVBench", "window_size", "Size of the entropy window", m_max_window_size);
    }

  private:
    size_t m_total_samples = 0;
    double m_sum_count_log_counter = 0.0;

    std::deque<float> m_internal_window; // Deque is better for pop_front
    size_t m_max_window_size = 299;      // Default NVBench window
  };
  struct RegressionResult {
    float slope_deg = 0.0f;
    float r_squared = 0.0f;
    bool is_valid = false;
    operator std::monostate() const {
      return std::monostate{};
    }
  };

  class EntropyRegression : public IStat<EntropyRegression, RegressionResult, EntropySlidingWindow> {
  public:
    [[nodiscard]] auto name() const -> std::string override {
      return "entropy_regression";
    }

    void calculate(RegressionResult &result, const EntropyWindowUpdate &update) override {

      if (update.was_sliding) {

        m_reg.slide_window(update.evicted_value, update.added_value);
      } else {
        const double x = static_cast<double>(update.current_size - 1);
        m_reg.update({x, static_cast<double>(update.added_value)});
      }

      const double raw_slope = m_reg.slope();
      const double r2 = m_reg.r_squared();

      if (std::isfinite(raw_slope) && std::isfinite(r2)) {
        result.is_valid = true;
        result.r_squared = static_cast<float>(r2);

        result.slope_deg = static_cast<float>(Utils::slope2deg(raw_slope));
      } else {
        result.is_valid = false;
        result.r_squared = 0.0f;
        result.slope_deg = 90.0f;
      }
    }

    [[nodiscard]] auto granularity() const -> MetricGranularity override {
      return MetricGranularity::EVERY_ELEMENT;
    }
    [[nodiscard]] auto saving_policy() const -> SavingPolicy override {
      return SavingPolicy::DISCARD;
    }
    [[nodiscard]] auto unit() const -> std::string override {
      return "";
    }

  private:
    Utils::online_linear_regression m_reg;
  };
} // namespace Baseliner::Stats
#endif