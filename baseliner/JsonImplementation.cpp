#include <baseliner/JsonImplementation.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <iomanip>
#include <ostream>
#include <variant>
namespace Baseliner {

  template <typename T>
  void serialize(std::ostream &oss, const T &obj) {
    json json_obj;
    json_obj = obj;
    oss << std::setw(2) << json_obj;
  }

  template void serialize<Metric>(std::ostream &oss, const Metric &obj);
  template void serialize<Result>(std::ostream &oss, const Result &obj);
  template void serialize<MetricData>(std::ostream &oss, const MetricData &obj);
  template void serialize<Option>(std::ostream &oss, const Option &obj);
  template void serialize<OptionsMap>(std::ostream &oss, const OptionsMap &obj);
  template void serialize<InterfaceOptions>(std::ostream &oss, const InterfaceOptions &obj);

  template void serialize<std::vector<Result>>(std::ostream &oss, const std::vector<Result> &obj);
  template void serialize<std::vector<Metric>>(std::ostream &oss, const std::vector<Metric> &obj);
  template void serialize<std::vector<Option>>(std::ostream &oss, const std::vector<Option> &obj);

  void to_json(json &json_obj, const Option &opt) {
    json_obj =
        json{{"description", opt.m_description}, {"value", opt.m_value}}; // We don't need the description in the JSON
    json_obj = opt.m_value;
  }
  void from_json(const json &json_obj, Option &opt) {
    // json_obj.at("description").get_to(opt.m_description);// Same as above
    json_obj.get_to(opt.m_value);
  }

  void to_json(json &json_obj, const Metric &metric) {
    if (!std::holds_alternative<std::monostate>(metric.m_data)) {
      json_obj["value"] = metric.m_data;
    }
    if (!metric.m_unit.empty()) {
      json_obj["unit"] = metric.m_unit;
    }
    json_obj["name"] = metric.m_name;
  }

  void to_json(json &json_obj, const Result &result) {
    json_obj["options"] = result.get_map();
    json_obj["kernel_name"] = result.get_kernel_name();
    json_obj["git_version"] = result.get_git_version();
    json_obj["baseliner_version"] = result.get_basliner_version();
    json_obj["execution_id"] = result.get_execution_uid();
    json_obj["datetime"] = result.get_date_time();
    json_obj["metrics"] = result.get_v_metrics();
  }
  void to_json(json &json_obj, const MetricData &metricData) {
    std::visit([&json_obj](auto &&arg) { json_obj = arg; }, metricData);
  }
} // namespace Baseliner