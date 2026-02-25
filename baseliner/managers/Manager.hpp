#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Recipe.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/GeneralStorage.hpp>
#include <functional>
#include <memory>
#include <sstream>
namespace Baseliner {
  struct OptionPreset {
    OptionsMap m_map;
    std::string m_description;
  };

  struct StatPreset {
    std::string m_description;
    std::vector<std::string> m_stats;
  };
  constexpr std::string_view DEFAULT_BENCHMARK = "Benchmark";
  constexpr std::string_view DEFAULT_STOPPING = "StoppingCriterion";

  class Manager {
  public:
    auto instance() -> Manager * {
      static Manager manager;
      return &manager;
    }

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

  private:
    std::unordered_map<std::string, IBackendStorage *> m_backends_storage;
    GeneralStatsStorage m_stats_storage;
    SuiteStorage m_suite_storage;
    StoppingCriterionStorage m_stopping_storage;

    // Preset Management item with IOptions
    std::unordered_map<std::string, std::unordered_map<std::string, OptionPreset>> m_case_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, OptionPreset>> m_benchmark_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, OptionPreset>> m_suite_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, OptionPreset>> m_stopping_presets;

    // Preset for stats (just names)
    std::unordered_map<std::string, StatPreset> m_stats_presets;

    // Defaults
    std::string m_default_benchmarl{DEFAULT_BENCHMARK};
    std::string m_default_stopping{DEFAULT_STOPPING};
  };
} // namespace Baseliner
#endif // BASELINER_MANAGER_HPP