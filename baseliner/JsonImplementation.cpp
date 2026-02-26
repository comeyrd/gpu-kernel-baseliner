#include <baseliner/JsonImplementation.hpp>
#include <baseliner/Metadata.hpp>
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
    oss << std::setw(1) << json_obj;
  }

  template void serialize<Metric>(std::ostream &oss, const Metric &obj);
  template void serialize<Result>(std::ostream &oss, const Result &obj);
  template void serialize<MetricData>(std::ostream &oss, const MetricData &obj);
  template void serialize<Option>(std::ostream &oss, const Option &obj);
  template void serialize<OptionsMap>(std::ostream &oss, const OptionsMap &obj);
  template void serialize<InterfaceOptions>(std::ostream &oss, const InterfaceOptions &obj);

  template void serialize<OptionPreset>(std::ostream &oss, const OptionPreset &obj);
  template void serialize<BackendMetadata>(std::ostream &oss, const BackendMetadata &obj);
  template void serialize<Ingredient>(std::ostream &oss, const Ingredient &obj);
  template void serialize<StatPreset>(std::ostream &oss, const StatPreset &obj);
  template void serialize<Metadata>(std::ostream &oss, const Metadata &obj);

  template void serialize<std::vector<Result>>(std::ostream &oss, const std::vector<Result> &obj);
  template void serialize<std::vector<Metric>>(std::ostream &oss, const std::vector<Metric> &obj);
  template void serialize<std::vector<Option>>(std::ostream &oss, const std::vector<Option> &obj);

  template void serialize<std::vector<Ingredient>>(std::ostream &oss, const std::vector<Ingredient> &obj);
  template void serialize<std::vector<StatPreset>>(std::ostream &oss, const std::vector<StatPreset> &obj);
  template void serialize<std::vector<BackendMetadata>>(std::ostream &oss, const std::vector<BackendMetadata> &obj);

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

  void to_json(json &json_obj, const OptionPreset &option_preset) {
    json_obj["name"] = option_preset.m_name;
    json_obj["description"] = option_preset.m_preset.m_description;
    json_obj["patch"] = option_preset.m_preset.m_map;
  }

  void to_json(json &json_obj, const StatPreset &stat_preset) {
    json_obj["name"] = stat_preset.m_name;
    json_obj["description"] = stat_preset.m_preset.m_description;
    json_obj["stats"] = stat_preset.m_preset.m_stats;
  }

  void to_json(json &json_obj, const BackendMetadata &backend_meta) {
    json_obj["name"] = backend_meta.m_name;
    json_obj["benchmarks"] = backend_meta.m_benchmaks;
    json_obj["cases"] = backend_meta.m_cases;
    json_obj["stats"] = backend_meta.m_stats;
  }
  void to_json(json &json_obj, const Ingredient &ingredient) {
    json_obj["name"] = ingredient.m_name;
    json_obj["presets"] = ingredient.m_presets;
  }
  void to_json(json &json_obj, const Metadata &metadata) {
    json_obj["baseliner_version"] = metadata.baseliner_version;
    json_obj["benchmarks"] = metadata.m_benchmarks;
    json_obj["cases"] = metadata.m_cases;
    json_obj["stopping"] = metadata.m_stopping_criterions;
    json_obj["suites"] = metadata.m_suites;
    json_obj["stats"] = metadata.m_general_stats;
    json_obj["stats_presets"] = metadata.m_stats_presets;
    json_obj["backends"] = metadata.m_backends;
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