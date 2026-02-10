#ifndef JSON_HANDLER_HPP
#define JSON_HANDLER_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
NLOHMANN_JSON_NAMESPACE_BEGIN
template <>
struct adl_serializer<Baseliner::float_milliseconds> {
  static void to_json(json &json_obj, const Baseliner::float_milliseconds &value) {
    json_obj = value.count();
  }

  static void from_json(const json &json_obj, Baseliner::float_milliseconds &value) {
    value = Baseliner::float_milliseconds(json_obj.get<float>());
  }
};
template <>
struct adl_serializer<Baseliner::MetricData> {
  static void to_json(json &json_obj, const Baseliner::MetricData &metricData) {

    std::visit([&json_obj](auto &&arg) { json_obj = arg; }, metricData);
  }
};
NLOHMANN_JSON_NAMESPACE_END

namespace Baseliner {

  // Define to_json and/or from_json for types you want to support
  void to_json(json &json_obj, const Option &opt);
  void from_json(const json &json_obj, Option &opt);

  void to_json(json &json_obj, const Result &result);
  void to_json(json &json_obj, const Metric &metric);
  void to_json(json &json_obj, const MetricStats &metricStats);
  template <typename T>
  void save_to_json(std::ostream &oss, T &obj) {
    json json_obj;
    to_json(json_obj, obj);
    oss << std::setw(2) << json_obj;
  }
} // namespace Baseliner

#endif // JSON_HANDLER_HPP