#ifndef BASELINER_METRIC_HPP
#define BASELINER_METRIC_HPP
#include <baseliner/Durations.hpp>

#include <string>
#include <variant>
#include <vector>

namespace Baseliner {

  using MetricData = std::variant<float_milliseconds, int64_t, std::string, float, std::vector<float_milliseconds>,
                                  std::vector<int64_t>, std::vector<std::string>, std::vector<float>>;

  struct MetricStats {
    std::string m_name;
    MetricData m_data;
    explicit MetricStats(std::string name)
        : m_name(name) {};
  };

  struct Metric {
    std::string m_name;
    std::string m_unit;
    MetricData m_data;
    std::vector<MetricStats> m_v_stats;

    Metric(std::string name, std::string unit, MetricData data)
        : m_name(name),
          m_unit(unit),
          m_data(data),
          m_v_stats() {};

    void add_stat(MetricStats &stat) {
      m_v_stats.push_back(stat);
    }
  };

} // namespace Baseliner
#endif // BASELINER_METRIC_HPP