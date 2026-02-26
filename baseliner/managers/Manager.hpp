#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/Metadata.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Recipe.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/GeneralStorage.hpp>
#include <functional>
#include <memory>
#include <sstream>
namespace Baseliner {

  constexpr std::string_view DEFAULT_BENCHMARK = "Benchmark";
  constexpr std::string_view DEFAULT_STOPPING = "StoppingCriterion";
  constexpr std::string_view DEFAULT_PRESET = "default";
  constexpr std::string_view DEFAULT_DESCRIPTION = "Default preset";

  enum ComponentType {
    NONE,
    CASE,
    BENCHMARK,
    SUITE,
    STOPPING,
  };

  static auto component_to_string(const ComponentType &type) -> std::string {
    switch (type) {
    case NONE:
      return "";
    case CASE:
      return "Case";
    case BENCHMARK:
      return "Benchmark";
    case SUITE:
      return "Suite";
    case STOPPING:
      return "Stopping";
    default:
      return "";
    }
  }

  class Manager {
  public:
    static auto instance() -> Manager * {
      static Manager manager;
      return &manager;
    }

    // TODO do same as in preset management ....
    auto build_recipe(const Recipe &recipe)
        -> std::variant<std::function<std::shared_ptr<IBenchmark>()>, std::function<std::shared_ptr<ISuite>()>> {
      auto backend_it = m_backends_storage.find(recipe.m_backend);
      auto case_it = m_case_presets.find(recipe.m_case.m_name);
      auto benchmark_it = m_benchmark_presets.find(recipe.m_benchmak.m_name);
      auto stopping_it = m_stopping_presets.find(recipe.m_stopping.m_name);
      if (backend_it == m_backends_storage.end()) {
        throw std::runtime_error("Baseliner Error : Backend " + recipe.m_backend + " not found");
      }
      if (case_it == m_case_presets.end()) {
        throw std::runtime_error("Baseliner Error : Case " + recipe.m_case.m_name + " not found");
      }
      if (benchmark_it == m_benchmark_presets.end()) {
        throw std::runtime_error("Baseliner Error : Case " + recipe.m_benchmak.m_name + " not found");
      }
      if (stopping_it == m_stopping_presets.end()) {
        throw std::runtime_error("Baseliner Error : Stopping " + recipe.m_stopping.m_name + " not found");
      }
      auto case_preset_it = case_it->second.find(recipe.m_case.m_preset);
      auto benchmark_preset_it = benchmark_it->second.find(recipe.m_benchmak.m_preset);
      auto stopping_preset_it = stopping_it->second.find(recipe.m_stopping.m_preset);
      auto stat_preset_it = m_stats_presets.find(recipe.m_stats.m_preset);
      if (case_preset_it == case_it->second.end()) {
        throw std::runtime_error("Baseliner Error : Preset " + recipe.m_case.m_preset + " not found for Case " +
                                 recipe.m_case.m_name);
      }
      if (benchmark_preset_it == benchmark_it->second.end()) {
        throw std::runtime_error("Baseliner Error : Preset " + recipe.m_benchmak.m_preset +
                                 " not found for Benchmark " + recipe.m_benchmak.m_name);
      }
      if (stopping_preset_it == stopping_it->second.end()) {
        throw std::runtime_error("Baseliner Error : Preset " + recipe.m_stopping.m_preset + " not found for Stopping " +
                                 recipe.m_stopping.m_name);
      }
      if (stat_preset_it == m_stats_presets.end()) {
        throw std::runtime_error("Baseliner Error : Preset " + recipe.m_stats.m_preset + " not found for Stat");
      }
      auto case_preset = case_preset_it->second;
      auto benchmark_preset = benchmark_preset_it->second;
      auto stopping_preset = stopping_preset_it->second;
      auto stat_preset = stat_preset_it->second;

      // Impl exists and preset exists

      auto *backend_storage = backend_it->second;
      // Adding General stats to general_stats
      // adding notfound_stats aka Device Specific stats to nofound_stats.
      std::vector<std::string> notfound_stats;
      std::vector<std::function<void(std::shared_ptr<Stats::StatsEngine>)>> general_stats;
      for (const auto &stat : stat_preset.m_stats) {
        if (m_stats_storage.has(stat)) {
          general_stats.push_back(m_stats_storage.at(stat));
        } else {
          notfound_stats.push_back(stat);
        }
      }

      auto benchmark_recipe = backend_storage->get_benchmark_with_case(
          recipe.m_benchmak.m_name, benchmark_preset.m_map, recipe.m_case.m_name, case_preset.m_map, notfound_stats);

      if (!m_stopping_storage.has(recipe.m_stopping.m_name)) {
        throw std::runtime_error("Baseliner Error : Stopping " + recipe.m_stopping.m_name + " not found in Storage");
      }
      auto stopping_criterion = m_stopping_storage.at(recipe.m_stopping.m_name);

      auto benchmark_with_stopping_and_stats = [benchmark_recipe, stopping_criterion,
                                                general_stats]() -> std::shared_ptr<IBenchmark> {
        auto bench = benchmark_recipe();
        bench->set_stopping_criterion(stopping_criterion);
        bench->add_stats(general_stats);
        return bench;
      };

      if (recipe.m_suite.has_value()) {
        auto suite = recipe.m_suite.value();
        auto suite_it = m_suite_presets.find(recipe.m_case.m_name);
        if (suite_it == m_suite_presets.end()) {
          throw std::runtime_error("Baseliner Error : Suite " + suite.m_name + " not found");
        }
        auto suite_preset_it = suite_it->second.find(suite.m_preset);
        if (suite_preset_it == suite_it->second.end()) {
          throw std::runtime_error("Baseliner Error : Preset " + suite.m_preset + " not found for Suite " +
                                   suite.m_name);
        }
        if (!m_suite_storage.has(suite.m_name)) {
          throw std::runtime_error("Baseliner Error : Suite " + suite.m_name + " not found in Storage");
        }
        auto suite_creator = m_suite_storage.at(suite.m_name);
        return [suite_creator, benchmark_with_stopping_and_stats]() -> std::shared_ptr<ISuite> {
          auto suite_ptr = suite_creator();
          suite_ptr->set_benchmark(benchmark_with_stopping_and_stats);
          return suite_ptr;
        };
      }
      return benchmark_with_stopping_and_stats;
    }

    /**
     ** list things
     **/

    auto list_cases_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> {
      return m_case_presets;
    }
    auto list_benchmark_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> {
      return m_benchmark_presets;
    }
    auto list_suite_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> {
      return m_suite_presets;
    }
    auto list_stopping_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> {
      return m_stopping_presets;
    }
    auto list_general_stats_presets() const -> std::unordered_map<std::string, InnerStatPreset> {
      return m_stats_presets;
    }
    auto list_backends() const -> std::unordered_map<std::string, IBackendStorage *> {
      return m_backends_storage;
    }
    auto list_stats() const -> std::vector<std::string> {
      return m_stats_storage.list();
    }
    /**
     ** Registering things
     **/
    void register_backend(const std::string &name, IBackendStorage *backend) {
      if (m_backends_storage.find(name) != m_backends_storage.end()) {
        throw std::runtime_error("Baseliner Error : Two backends with the same name registered : " + name);
      }
      m_backends_storage[name] = backend;
    }

    void register_general_stat(const std::string &name,
                               const std::function<void(std::shared_ptr<Stats::StatsEngine>)> &stat_factory) {
      m_stats_storage.insert(name, stat_factory);
    }
    void register_suite(const std::string &name, const std::function<std::shared_ptr<ISuite>()> &suite_factory) {
      m_suite_storage.insert(name, suite_factory);
      add_preset(name, std::string(DEFAULT_PRESET),
                 InnerOptionPreset{std::string(DEFAULT_DESCRIPTION), suite_factory()->gather_options()},
                 ComponentType::SUITE);
    }
    void register_stopping(const std::string &name,
                           const std::function<std::unique_ptr<StoppingCriterion>()> &stopping_factory) {
      m_stopping_storage.insert(name, stopping_factory);
      add_preset(name, std::string(DEFAULT_PRESET),
                 InnerOptionPreset{std::string(DEFAULT_DESCRIPTION), stopping_factory()->gather_options()},
                 ComponentType::STOPPING);
    }

    /**
     ** Registring Preset
     **/
    void add_preset(const std::string &name, const std::string &preset_name, const InnerOptionPreset &preset,
                    ComponentType type) {
      std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> *map = nullptr;
      switch (type) {
      case NONE: {
        throw std::runtime_error("add_preset called with NONE ComponentType");
      }
      case CASE:
        map = &m_case_presets;
        break;
      case BENCHMARK:
        map = &m_benchmark_presets;
        break;
      case SUITE:
        map = &m_suite_presets;
        break;
      case STOPPING:
        map = &m_stopping_presets;
        break;
      }
      auto &specific = (*map)[name];
      if (specific.find(preset_name) != specific.end() && preset_name != std::string(DEFAULT_PRESET)) {
        throw std::runtime_error("Baseliner Error : Preset " + preset_name +
                                 " already defined for : " + component_to_string(type) + " : " + name);
      }
      specific[preset_name] = preset;
    }
    auto generate_metadata() -> Metadata {
      Metadata metadata{};
      for (const auto &[name, map] : m_benchmark_presets) {
        Ingredient ingredient{name, {}};
        for (const auto &[name, preset] : map) {
          ingredient.m_presets.push_back({name, preset});
        }
        metadata.m_benchmarks.push_back(ingredient);
      }
      for (const auto &[name, map] : m_case_presets) {
        Ingredient ingredient{name, {}};
        for (const auto &[name, preset] : map) {
          ingredient.m_presets.push_back({name, preset});
        }
        metadata.m_cases.push_back(ingredient);
      }
      for (const auto &[name, map] : m_stopping_presets) {
        Ingredient ingredient{name, {}};
        for (const auto &[name, preset] : map) {
          ingredient.m_presets.push_back({name, preset});
        }
        metadata.m_stopping_criterions.push_back(ingredient);
      }
      for (const auto &[name, map] : m_suite_presets) {
        Ingredient ingredient{name, {}};
        for (const auto &[name, preset] : map) {
          ingredient.m_presets.push_back({name, preset});
        }
        metadata.m_suites.push_back(ingredient);
      }
      for (const auto &[name, preset] : m_stats_presets) {
        StatPreset ingredient{name, preset};
        metadata.m_stats_presets.push_back(ingredient);
      }
      metadata.m_general_stats = m_stats_storage.list();
      for (auto [_, backend] : m_backends_storage) {
        metadata.m_backends.push_back(backend->generate_backend_metadata());
      }
      return metadata;
    };

  private:
    std::unordered_map<std::string, IBackendStorage *> m_backends_storage;
    GeneralStatsStorage m_stats_storage;
    SuiteStorage m_suite_storage;
    StoppingCriterionStorage m_stopping_storage;

    // Preset Management item with IOptions
    std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> m_case_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> m_benchmark_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> m_suite_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerOptionPreset>> m_stopping_presets;

    // Preset for stats (just names)
    std::unordered_map<std::string, InnerStatPreset> m_stats_presets{
        {std::string(DEFAULT_PRESET), {"Only Median", {"Median"}}}};
    // Defaults
    std::string m_default_benchmarl = std::string(DEFAULT_BENCHMARK);
    std::string m_default_stopping = std::string(DEFAULT_STOPPING);
  };
} // namespace Baseliner
#endif // BASELINER_MANAGER_HPP