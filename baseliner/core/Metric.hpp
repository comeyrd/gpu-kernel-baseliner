#ifndef BASELINER_CORE_METRIC_HPP
#define BASELINER_CORE_METRIC_HPP
#include <baseliner/core/Durations.hpp>

#include <baseliner/core/stats/StatsType.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace Baseliner {

  using MetricData = std::variant<std::monostate, float_milliseconds, int64_t, std::string, int, size_t, float,
                                  std::vector<float_milliseconds>, std::vector<int>, std::vector<int64_t>,
                                  std::vector<std::string>, std::vector<float>, ConfidenceInterval<float>,
                                  ConfidenceInterval<float_milliseconds>, ConfidenceInterval<size_t>>;

  DESCRIBE_ENUM(MetricGranularity, ENUM_VALUE(EVERY_ELEMENT), ENUM_VALUE(EVERY_BATCH), ENUM_VALUE(ON_DEMAND),
                ENUM_VALUE(ONCE))

  struct Metric {
    std::string name;
    std::string unit;
    MetricData data;
    MetricGranularity granularity;
  };
  DESCRIBE(Metric, FIELD(name), FIELD(unit), FIELD(data), FIELD(granularity))

} // namespace Baseliner
#endif // BASELINER_METRIC_HPP