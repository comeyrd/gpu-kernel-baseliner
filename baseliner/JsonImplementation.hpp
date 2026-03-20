#ifndef JSON_IMPLEMENTATION_HPP
#define JSON_IMPLEMENTATION_HPP
#include <baseliner/AxeSweeping.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Output.hpp>
#include <baseliner/hardware/Backend.hpp>
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
  NLOHMANN_JSON_SERIALIZE_ENUM(OnIncompatible, {
                                                   {OnIncompatible::Skip, "skip"},
                                                   {OnIncompatible::Error, "error"},
                                               })

  NLOHMANN_JSON_SERIALIZE_ENUM(SweepStrategy, {
                                                  {SweepStrategy::Carthesian, "carthesian"},
                                              })

  NLOHMANN_JSON_SERIALIZE_ENUM(SweepPolicy, {
                                                {SweepPolicy::PowersOfTwo, "powers_of_two"},
                                                {SweepPolicy::LinearRange, "linear_range"},
                                                {SweepPolicy::Enumerated, "enumerated"},
                                            })

  // Define to_json and/or from_json for types you want to support

  void to_json(json &json_obj, const Metric &metric);
  namespace Hardware {
    void to_json(json &json_obj, const HardwareInfo &device);
  }
  void to_json(json &json_obj, const Option &opt);
  void from_json(const json &json_obj, Option &opt);
  void from_json(const json &json_obj, HardwareInfo &device);
  // Output
  void to_json(json &json_obj, const PlannedComponent &component);
  void from_json(const json &json_obj, PlannedComponent &component);

  void to_json(json &json_obj, const PlannedStat &stat);
  void from_json(const json &json_obj, PlannedStat &stat);

  void to_json(json &json_obj, const Plan &plan);
  void from_json(const json &json_obj, Plan &plan);

  void to_json(json &json_obj, const SingleRunReport &report);
  void from_json(const json &json_obj, SingleRunReport &report);

  void to_json(json &json_obj, const BenchmarkReport &report);
  void from_json(const json &json_obj, BenchmarkReport &report);

  void to_json(json &json_obj, const RunReport &report);
  void from_json(const json &json_obj, RunReport &report);

  void to_json(json &json_obj, const Report &report);
  void from_json(const json &json_obj, Report &report);

  void to_json(json &json_obj, const RecipeComponent &component);
  void from_json(const json &json_obj, RecipeComponent &component);

  void to_json(json &json_obj, const RecipeStat &stat);
  void from_json(const json &json_obj, RecipeStat &stat);

  void to_json(json &json_obj, const SweepHint &hint);
  void from_json(const json &json_obj, SweepHint &hint);

  void to_json(json &json_obj, const SweepAxis &axis);
  void from_json(const json &json_obj, SweepAxis &axis);

  void to_json(json &json_obj, const ResolvedAxis &axis);
  void from_json(const json &json_obj, ResolvedAxis &axis);

  void to_json(json &json_obj, const SweepSpec &sweep);
  void from_json(const json &json_obj, SweepSpec &sweep);

  void to_json(json &json_obj, const Recipe &recipe);
  void from_json(const json &json_obj, Recipe &recipe);

  void to_json(json &json_obj, const CampaignOverrides &overrides);
  void from_json(const json &json_obj, CampaignOverrides &overrides);

  void to_json(json &json_obj, const Campaign &campaign);
  void from_json(const json &json_obj, Campaign &campaign);

  void to_json(json &json_obj, const Protocol &protocol);
  void from_json(const json &json_obj, Protocol &protocol);

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