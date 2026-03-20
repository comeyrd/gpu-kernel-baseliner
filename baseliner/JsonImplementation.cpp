#include <baseliner/JsonImplementation.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>

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
  template <typename T>
  void de_serialize(std::istream &iss, T &obj) {
    json json_obj;
    iss >> json_obj;
    json_obj.get_to(obj);
  }

  template void serialize<Metric>(std::ostream &oss, const Metric &obj);
  template void serialize<MetricData>(std::ostream &oss, const MetricData &obj);
  template void serialize<Option>(std::ostream &oss, const Option &obj);
  template void serialize<OptionsMap>(std::ostream &oss, const OptionsMap &obj);
  template void serialize<InterfaceOptions>(std::ostream &oss, const InterfaceOptions &obj);

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

  namespace Hardware {
    void to_json(json &json_obj, const HardwareInfo &device) {
      json_obj["name"] = device.name;
    }
    void from_json(const json &json_obj, HardwareInfo &device) {
      json_obj.at("name").get_to(device.name);
    }
  } // namespace Hardware
  void to_json(json &json_obj, const MetricData &metricData) {
    std::visit([&json_obj](auto &&arg) { json_obj = arg; }, metricData);
  }
  void to_json(json &json_obj, const PlannedComponent &component) {
    json_obj["impl"] = component.m_impl;
    json_obj["preset"] = component.m_preset;
    json_obj["options"] = component.m_options;
  }
  void from_json(const json &json_obj, PlannedComponent &component) {
    json_obj.at("impl").get_to(component.m_impl);
    json_obj.at("preset").get_to(component.m_preset);
    json_obj.at("options").get_to(component.m_options);
  }

  void to_json(json &json_obj, const PlannedStat &stat) {
    json_obj["preset"] = stat.m_preset;
    json_obj["stats"] = stat.m_stats;
    json_obj["options"] = stat.m_options;
  }
  void from_json(const json &json_obj, PlannedStat &stat) {
    json_obj.at("preset").get_to(stat.m_preset);
    json_obj.at("stats").get_to(stat.m_stats);
    json_obj.at("options").get_to(stat.m_options);
  }

  void to_json(json &json_obj, const Plan &plan) {
    json_obj["campaign_name"] = plan.m_campaign_name;
    json_obj["recipe_name"] = plan.m_recipe_name;
    json_obj["case"] = plan.m_case;
    json_obj["backend"] = plan.m_backend;
    json_obj["benchmark"] = plan.m_benchmark;
    json_obj["stopping"] = plan.m_stopping;
    json_obj["stats"] = plan.m_stats;
    json_obj["on_incompatible"] = plan.m_on_incompatible;
    if (plan.m_sweep.has_value()) {
      json_obj["sweep"] = plan.m_sweep.value();
    }
  }
  void from_json(const json &json_obj, Plan &plan) {
    json_obj.at("campaign_name").get_to(plan.m_campaign_name);
    json_obj.at("recipe_name").get_to(plan.m_recipe_name);
    json_obj.at("case").get_to(plan.m_case);
    json_obj.at("backend").get_to(plan.m_backend);
    json_obj.at("benchmark").get_to(plan.m_benchmark);
    json_obj.at("stopping").get_to(plan.m_stopping);
    json_obj.at("stats").get_to(plan.m_stats);
    json_obj.at("on_incompatible").get_to(plan.m_on_incompatible);
    if (json_obj.contains("sweep")) {
      plan.m_sweep = json_obj.at("sweep").get<SweepSpec>();
    } else {
      plan.m_sweep = {};
    }
  }

  void to_json(json &json_obj, const SingleRunReport &report) {
    if (report.m_sweep_point.has_value()) {
      json_obj["sweep_point"] = report.m_sweep_point.value();
    }
    json_obj["measurements"] = report.m_measurements;
  }
  void from_json(const json &json_obj, SingleRunReport &report) {
    if (json_obj.contains("sweep_point")) {
      report.m_sweep_point = json_obj.at("sweep_point").get<OptionsMap>();
    } else {
      report.m_sweep_point = {};
    }
    // json_obj.at("measurements").get_to(report.m_measurements); //TODO measurements
  }

  void to_json(json &json_obj, const BenchmarkReport &report) {
    json_obj["results"] = report.m_results;
    json_obj["hardware"] = report.m_hardware;
  }
  void from_json(const json &json_obj, BenchmarkReport &report) {
    json_obj.at("results").get_to(report.m_results);
    json_obj.at("hardware").get_to<Hardware::HardwareInfo>(report.m_hardware);
  }

  void to_json(json &json_obj, const RunReport &report) {
    json_obj["plan"] = report.m_plan;
    json_obj["benchmark_report"] = report.m_benchmark_report;
  }
  void from_json(const json &json_obj, RunReport &report) {
    json_obj.at("plan").get_to(report.m_plan);
    json_obj.at("benchmark_report").get_to(report.m_benchmark_report);
  }

  void to_json(json &json_obj, const Report &report) {
    json_obj["baseliner_version"] = report.m_baseliner_version;
    json_obj["git_version"] = report.m_git_version;
    json_obj["datetime"] = report.m_datetime;
    json_obj["runs"] = report.m_runs;
  }
  void from_json(const json &json_obj, Report &report) {
    json_obj.at("baseliner_version").get_to(report.m_baseliner_version);
    json_obj.at("git_version").get_to(report.m_git_version);
    json_obj.at("datetime").get_to(report.m_datetime);
    json_obj.at("runs").get_to(report.m_runs);
  }

  void to_json(json &json_obj, const RecipeComponent &component) {
    json_obj["impl"] = component.m_impl;
    if (component.m_preset.has_value()) {
      json_obj["preset"] = component.m_preset.value();
    }
  }
  void from_json(const json &json_obj, RecipeComponent &component) {
    json_obj.at("impl").get_to(component.m_impl);
    if (json_obj.contains("preset")) {
      component.m_preset = json_obj.at("preset").get<std::string>();
    } else {
      component.m_preset = {};
    }
  }

  void to_json(json &json_obj, const RecipeStat &stat) {
    json_obj["preset"] = stat.m_preset;
  }
  void from_json(const json &json_obj, RecipeStat &stat) {
    json_obj.at("preset").get_to(stat.m_preset);
  }

  void to_json(json &json_obj, const SweepHint &hint) {
    json_obj["policy"] = hint.m_policy;
    json_obj["min"] = hint.m_min;
    json_obj["max"] = hint.m_max;
    json_obj["step"] = hint.m_step;
    json_obj["enumerated"] = hint.m_enumerated;
  }
  void from_json(const json &json_obj, SweepHint &hint) {
    json_obj.at("policy").get_to(hint.m_policy);
    json_obj.at("min").get_to(hint.m_min);
    json_obj.at("max").get_to(hint.m_max);
    json_obj.at("step").get_to(hint.m_step);
    json_obj.at("enumerated").get_to(hint.m_enumerated);
  }

  void to_json(json &json_obj, const SweepAxis &axis) {
    json_obj["interface"] = axis.m_interface;
    json_obj["option"] = axis.m_option;
    json_obj["hint"] = axis.m_hint;
  }
  void from_json(const json &json_obj, SweepAxis &axis) {
    json_obj.at("interface").get_to(axis.m_interface);
    json_obj.at("option").get_to(axis.m_option);
    json_obj.at("hint").get_to(axis.m_hint);
  }

  void to_json(json &json_obj, const ResolvedAxis &axis) {
    json_obj["interface"] = axis.m_interface;
    json_obj["option"] = axis.m_option;
    json_obj["value"] = axis.value;
  }
  void from_json(const json &json_obj, ResolvedAxis &axis) {
    json_obj.at("interface").get_to(axis.m_interface);
    json_obj.at("option").get_to(axis.m_option);
    json_obj.at("value").get_to(axis.value);
  }

  void to_json(json &json_obj, const SweepSpec &sweep) {
    json_obj["strategy"] = sweep.m_strategy;
    json_obj["axes"] = sweep.m_axes;
  }
  void from_json(const json &json_obj, SweepSpec &sweep) {
    json_obj.at("strategy").get_to(sweep.m_strategy);
    json_obj.at("axes").get_to(sweep.m_axes);
  }

  void to_json(json &json_obj, const Recipe &recipe) {
    json_obj["description"] = recipe.m_description;
    if (recipe.m_benchmark.has_value()) {
      json_obj["benchmark"] = recipe.m_benchmark.value();
    }
    if (recipe.m_stopping.has_value()) {
      json_obj["stopping"] = recipe.m_stopping.value();
    }
    if (recipe.m_stats.has_value()) {
      json_obj["stats"] = recipe.m_stats.value();
    }
    if (recipe.m_sweep.has_value()) {
      json_obj["sweep"] = recipe.m_sweep.value();
    }
  }
  void from_json(const json &json_obj, Recipe &recipe) {
    json_obj.at("description").get_to(recipe.m_description);
    if (json_obj.contains("benchmark")) {
      recipe.m_benchmark = json_obj.at("benchmark").get<RecipeComponent>();
    } else {
      recipe.m_benchmark = {};
    }
    if (json_obj.contains("stopping")) {
      recipe.m_stopping = json_obj.at("stopping").get<RecipeComponent>();
    } else {
      recipe.m_stopping = {};
    }
    if (json_obj.contains("stats")) {
      recipe.m_stats = json_obj.at("stats").get<RecipeStat>();
    } else {
      recipe.m_stats = {};
    }
    if (json_obj.contains("sweep")) {
      recipe.m_sweep = json_obj.at("sweep").get<SweepSpec>();
    } else {
      recipe.m_sweep = {};
    }
  }

  void to_json(json &json_obj, const CampaignOverrides &overrides) {
    if (overrides.m_benchmark.has_value()) {
      json_obj["benchmark"] = overrides.m_benchmark.value();
    }
    if (overrides.m_stopping.has_value()) {
      json_obj["stopping"] = overrides.m_stopping.value();
    }
    if (overrides.m_stats.has_value()) {
      json_obj["stats"] = overrides.m_stats.value();
    }
  }
  void from_json(const json &json_obj, CampaignOverrides &overrides) {
    if (json_obj.contains("benchmark")) {
      overrides.m_benchmark = json_obj.at("benchmark").get<RecipeComponent>();
    } else {
      overrides.m_benchmark = {};
    }
    if (json_obj.contains("stopping")) {
      overrides.m_stopping = json_obj.at("stopping").get<RecipeComponent>();
    } else {
      overrides.m_stopping = {};
    }
    if (json_obj.contains("stats")) {
      overrides.m_stats = json_obj.at("stats").get<RecipeStat>();
    } else {
      overrides.m_stats = {};
    }
  }

  void to_json(json &json_obj, const Campaign &campaign) {
    json_obj["name"] = campaign.m_name;
    json_obj["recipe"] = campaign.m_recipe;
    json_obj["cases"] = campaign.m_cases;
    json_obj["backends"] = campaign.m_backends;
    json_obj["overrides"] = campaign.m_overrides;
    json_obj["on_incompatible"] = campaign.m_on_incompatible;
  }
  void from_json(const json &json_obj, Campaign &campaign) {
    json_obj.at("name").get_to(campaign.m_name);
    json_obj.at("recipe").get_to(campaign.m_recipe);
    json_obj.at("cases").get_to(campaign.m_cases);
    json_obj.at("backends").get_to(campaign.m_backends);
    json_obj.at("overrides").get_to(campaign.m_overrides);
    json_obj.at("on_incompatible").get_to(campaign.m_on_incompatible);
  }

  void to_json(json &json_obj, const Protocol &protocol) {
    json_obj["baseliner_version"] = protocol.m_baseliner_version;
    json_obj["presets"] = protocol.m_presets;
    json_obj["stats_presets"] = protocol.m_stats_presets;
    json_obj["recipes"] = protocol.m_recipes;
    json_obj["campaigns"] = protocol.m_campaigns;
  }
  void from_json(const json &json_obj, Protocol &protocol) {
    json_obj.at("baseliner_version").get_to(protocol.m_baseliner_version);
    json_obj.at("presets").get_to(protocol.m_presets);
    json_obj.at("stats_presets").get_to(protocol.m_stats_presets);
    json_obj.at("recipes").get_to(protocol.m_recipes);
    json_obj.at("campaigns").get_to(protocol.m_campaigns);
  }
} // namespace Baseliner