#include "baseliner/managers/Manager.hpp"
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
  template <typename T>
  void de_serialize(std::istream &iss, T &obj) {
    json json_obj;
    iss >> json_obj;
    json_obj.get_to(obj);
  }

  template void serialize<Metric>(std::ostream &oss, const Metric &obj);
  template void serialize<Result>(std::ostream &oss, const Result &obj);
  template void serialize<MetricData>(std::ostream &oss, const MetricData &obj);
  template void serialize<Option>(std::ostream &oss, const Option &obj);
  template void serialize<OptionsMap>(std::ostream &oss, const OptionsMap &obj);
  template void serialize<InterfaceOptions>(std::ostream &oss, const InterfaceOptions &obj);
  template void serialize<Config>(std::ostream &oss, const Config &obj);
  template void serialize<PresetDefinition>(std::ostream &oss, const PresetDefinition &obj);
  template void serialize<Recipe>(std::ostream &oss, const Recipe &obj);
  template void serialize<WithPreset>(std::ostream &oss, const WithPreset &obj);

  template void serialize<BackendMetadata>(std::ostream &oss, const BackendMetadata &obj);
  template void serialize<Ingredient>(std::ostream &oss, const Ingredient &obj);
  template void serialize<Preset>(std::ostream &oss, const Preset &obj);
  template void serialize<Metadata>(std::ostream &oss, const Metadata &obj);

  template void serialize<std::vector<Result>>(std::ostream &oss, const std::vector<Result> &obj);
  template void serialize<std::vector<Metric>>(std::ostream &oss, const std::vector<Metric> &obj);
  template void serialize<std::vector<Option>>(std::ostream &oss, const std::vector<Option> &obj);

  template void serialize<std::vector<Ingredient>>(std::ostream &oss, const std::vector<Ingredient> &obj);
  template void serialize<std::vector<Preset>>(std::ostream &oss, const std::vector<Preset> &obj);
  template void serialize<std::vector<BackendMetadata>>(std::ostream &oss, const std::vector<BackendMetadata> &obj);
  template void serialize<std::vector<Recipe>>(std::ostream &oss, const std::vector<Recipe> &obj);

  template void de_serialize<std::vector<Recipe>>(std::istream &iss, std::vector<Recipe> &obj);
  template void de_serialize<WithPreset>(std::istream &iss, WithPreset &obj);
  template void de_serialize<Recipe>(std::istream &iss, Recipe &obj);
  template void de_serialize<Config>(std::istream &iss, Config &obj);
  template void de_serialize<PresetDefinition>(std::istream &iss, PresetDefinition &obj);
  template void de_serialize<std::vector<PresetDefinition>>(std::istream &iss, std::vector<PresetDefinition> &obj);
  template void de_serialize<std::variant<OptionsMap, std::vector<std::string>>>(
      std::istream &iss, std::variant<OptionsMap, std::vector<std::string>> &obj);

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

  void to_json(json &json_obj, const Preset &option_preset) {
    json_obj["name"] = option_preset.m_name;
    json_obj["description"] = option_preset.m_preset.m_description;
    json_obj["options"] = option_preset.m_preset.m_options;
  }
  void to_json(json &json_obj, const std::variant<OptionsMap, std::vector<std::string>> &option_preset) {
    if (std::holds_alternative<std::vector<std::string>>(option_preset)) {
      json_obj = std::get<std::vector<std::string>>(option_preset);
    } else {
      json_obj = std::get<OptionsMap>(option_preset);
    }
  }

  void from_json(const json &json_obj, std::variant<OptionsMap, std::vector<std::string>> &opt) {
    if (json_obj.is_array()) {
      opt = json_obj.get<std::vector<std::string>>();
    } else if (json_obj.is_object()) {
      opt = json_obj.get<OptionsMap>();
    } else {
      throw std::invalid_argument("Invalid JSON structure for option_preset: expected an array or an object.");
    }
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
    json_obj["stopping_criterion"] = metadata.m_stopping_criterions;
    json_obj["suites"] = metadata.m_suites;
    json_obj["stats"] = metadata.m_general_stats;
    json_obj["stats_presets"] = metadata.m_stats_presets;
    json_obj["backends"] = metadata.m_backends;
  }

  void to_json(json &json_obj, const Result &result) {
    json_obj["baseliner_version"] = result.get_basliner_version();
    json_obj["git_version"] = result.get_git_version();
    json_obj["execution_id"] = result.get_execution_uid();
    json_obj["datetime"] = result.get_date_time();
    json_obj["kernel_name"] = result.get_kernel_name();
    json_obj["metrics"] = result.get_v_metrics();
    json_obj["options"] = result.get_map();
  }
  void to_json(json &json_obj, const MetricData &metricData) {
    std::visit([&json_obj](auto &&arg) { json_obj = arg; }, metricData);
  }
  void to_json(json &json_obj, const Config &config) {
    json_obj["baseliner_version"] = config.m_baseliner_version;
    json_obj["presets"] = config.m_presets;
    json_obj["recipes"] = config.m_recipes;
  }
  void from_json(const json &json_obj, Config &config) {
    json_obj.at("baseliner_version").get_to(config.m_baseliner_version);
    json_obj.at("presets").get_to(config.m_presets);
    json_obj.at("recipes").get_to(config.m_recipes);
  }

  void to_json(json &json_obj, const PresetDefinition &preset) {
    json_obj["implementation"] = preset.m_implementation_name;
    json_obj["preset_name"] = preset.m_preset_name;
    json_obj["description"] = preset.m_description;
    json_obj["options"] = preset.m_options;
  }
  void from_json(const json &json_obj, PresetDefinition &preset) {
    json_obj.at("implementation").get_to(preset.m_implementation_name);
    json_obj.at("preset_name").get_to(preset.m_preset_name);
    if (json_obj.contains("description")) {
      json_obj.at("description").get_to(preset.m_description);
    } else {
      preset.m_description = std::string(DEFAULT_DESCRIPTION);
    }
    json_obj.at("options").get_to(preset.m_options);
  }

  void to_json(json &json_obj, const Recipe &recipe) {
    json_obj["backend"] = recipe.m_backend;
    json_obj["benchmark"] = recipe.m_benchmak;
    json_obj["case"] = recipe.m_case;
    json_obj["stats"] = recipe.m_stats;
    json_obj["stopping_criterion"] = recipe.m_stopping;
    if (recipe.m_suite) {
      json_obj["suite"] = recipe.m_suite.value();
    }
  }
  void from_json(const json &json_obj, Recipe &recipe) {
    json_obj.at("backend").get_to(recipe.m_backend);
    json_obj.at("case").get_to(recipe.m_case);
    if (json_obj.contains("benchmark")) {
      json_obj.at("benchmark").get_to(recipe.m_benchmak);
    } else {
      recipe.m_benchmak = {std::string(DEFAULT_BENCHMARK), std::string(DEFAULT_PRESET)};
    }
    if (json_obj.contains("stats")) {
      json_obj.at("stats").get_to(recipe.m_stats);
    } else {
      recipe.m_stats = {component_to_string(ComponentType::STAT), std::string(DEFAULT_PRESET)};
    }
    if (json_obj.contains("stopping_criterion")) {
      json_obj.at("stopping_criterion").get_to(recipe.m_stopping);
    } else {
      recipe.m_stopping = {std::string(DEFAULT_STOPPING), std::string(DEFAULT_PRESET)};
    }
    if (json_obj.contains("suite")) {
      WithPreset temp;
      json_obj.at("suite").get_to(temp);
      recipe.m_suite = temp;
    }
  }
  void to_json(json &json_obj, const WithPreset &w_preset) {
    json_obj["name"] = w_preset.m_name;
    json_obj["preset"] = w_preset.m_preset;
  }
  void from_json(const json &json_obj, WithPreset &w_preset) {
    if (json_obj.is_string()) {
      json_obj.get_to(w_preset.m_name);
      w_preset.m_preset = std::string(DEFAULT_PRESET);
    } else {
      json_obj.at("name").get_to(w_preset.m_name);
      if (json_obj.contains("preset")) {
        json_obj.at("preset").get_to(w_preset.m_preset);
      } else {
        w_preset.m_name = std::string(DEFAULT_PRESET);
      }
    }
  }

} // namespace Baseliner