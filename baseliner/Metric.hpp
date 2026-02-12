#ifndef BASELINER_METRIC_HPP
#define BASELINER_METRIC_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/StatsType.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace Baseliner {

  using MetricData =
      std::variant<std::monostate, float_milliseconds, int64_t, std::string, size_t, float,
                   std::vector<float_milliseconds>, std::vector<int64_t>, std::vector<std::string>, std::vector<float>,
                   ConfidenceInterval<float>, ConfidenceInterval<float_milliseconds>, ConfidenceInterval<size_t>>;

  struct MetricStats {
    std::string m_name;
    MetricData m_data;
  };

  struct Metric {
    std::string m_name;
    std::string m_unit;
    MetricData m_data;
    std::vector<MetricStats> m_v_stats;
  };

} // namespace Baseliner
#endif // BASELINER_METRIC_HPP