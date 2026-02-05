#include <baseliner/JsonHandler.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
namespace Baseliner {

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
    j["name"] = metric.m_name;
    j["unit"] = metric.m_unit;
    j["value"] = metric.m_data;
    j["stats"] = metric.m_v_stats;
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