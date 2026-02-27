#ifndef JSON_IMPLEMENTATION_HPP
#define JSON_IMPLEMENTATION_HPP
#include <baseliner/ConfigFile.hpp>
#include <baseliner/Metadata.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/stats/StatsType.hpp>
#include <iomanip>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;
namespace Baseliner {
  void to_json(json &json_obj, const MetricData &metricData);

  template <typename T>
  void to_json(json &json_obj, const ConfidenceInterval<T> &obj);
} // namespace Baseliner

NLOHMANN_JSON_NAMESPACE_BEGIN
template <>
struct adl_serializer<Baseliner::float_milliseconds> {
  static void to_json(ordered_json &json_obj, const Baseliner::float_milliseconds &value) {
    json_obj = value.count();
  }

  static void from_json(const ordered_json &json_obj, Baseliner::float_milliseconds &value) {
    value = Baseliner::float_milliseconds(json_obj.get<float>());
  }
};
template <>
struct adl_serializer<Baseliner::MetricData> {
  static void to_json(ordered_json &json_obj, const Baseliner::MetricData &metricData) {
    Baseliner::to_json(json_obj, metricData);
  }
};
template <>
struct adl_serializer<std::monostate> {
  static void to_json(ordered_json &json_obj, const std::monostate & /*monostate*/) {
    json_obj = "";
  }
};
NLOHMANN_JSON_NAMESPACE_END

namespace Baseliner {

  // Define to_json and/or from_json for types you want to support
  void to_json(json &json_obj, const Option &opt);
  void from_json(const json &json_obj, Option &opt);

  void to_json(json &json_obj, const Result &result);
  void to_json(json &json_obj, const Metric &metric);

  void to_json(json &json_obj, const BenchmarkResult &result);
  void to_json(json &json_obj, const RunResult &result);

  void to_json(json &json_obj, const Preset &option_preset);

  void to_json(json &json_obj, const BackendMetadata &backend_meta);
  void to_json(json &json_obj, const Ingredient &ingredient);
  void to_json(json &json_obj, const Metadata &metadata);

  void to_json(json &json_obj, const Config &config);
  void from_json(const json &json_obj, Config &config);
  void to_json(json &json_obj, const PresetDefinition &preset);
  void from_json(const json &json_obj, PresetDefinition &preset);

  void to_json(json &json_obj, const Recipe &recipe);
  void from_json(const json &json_obj, Recipe &recipe);
  void to_json(json &json_obj, const std::variant<OptionsMap, std::vector<std::string>> &option_preset);
  void from_json(const json &json_obj, std::variant<OptionsMap, std::vector<std::string>> &opt);
  void to_json(json &json_obj, const WithPreset &w_preset);
  void from_json(const json &json_obj, WithPreset &w_preset);

  template <typename T>
  void to_json(json &json_obj, const ConfidenceInterval<T> &obj) {
    json_obj["high"] = obj.high;
    json_obj["low"] = obj.low;
  };

  template <typename T>
  void save_to_json(std::ostream &oss, T &obj) {
    json json_obj;
    to_json(json_obj, obj);
    oss << std::setw(2) << json_obj;
  }
} // namespace Baseliner

#endif // JSON_MANAGER_HPP