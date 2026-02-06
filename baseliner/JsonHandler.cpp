#include <baseliner/JsonHandler.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &os, const T &obj) {
    json j;
    j = obj;
    os << std::setw(2) << j;
  }

  template void serialize<Metric>(std::ostream &os, const Metric &obj);
  template void serialize<Result>(std::ostream &os, const Result &obj);
  template void serialize<MetricData>(std::ostream &os, const MetricData &obj);
  template void serialize<MetricStats>(std::ostream &os, const MetricStats &obj);
  template void serialize<Option>(std::ostream &os, const Option &obj);

  template void serialize<std::vector<Result>>(std::ostream &os, const std::vector<Result> &obj);
  template void serialize<std::vector<Metric>>(std::ostream &os, const std::vector<Metric> &obj);

  void to_json(json &j, const Option &opt) {
    j = json{{"description", opt.m_description}, {"value", opt.m_value}}; // We don't need the description in the JSON
    j = opt.m_value;
  }
  void from_json(const json &j, Option &opt) {
    // j.at("description").get_to(opt.m_description);// Same as above
    j.get_to(opt.m_value);
  }

  void to_json(json &j, const MetricStats &metricStats) {
    j["name"] = metricStats.m_name;
    j["value"] = metricStats.m_data;
  };

  void to_json(json &j, const Metric &metric) {
    j["stats"] = metric.m_v_stats;
    j["value"] = metric.m_data;
    j["unit"] = metric.m_unit;
    j["name"] = metric.m_name;
  }

  void to_json(json &j, const Result &result) {
    j["options"] = result.m_map;
    j["kernel_name"] = result.m_kernel_name;
    j["git_version"] = result.m_git_version;
    j["execution_id"] = result.m_execution_uid;
    j["datetime"] = result.m_date_time;
    j["metrics"] = result.m_v_metrics;
  }

} // namespace Baseliner