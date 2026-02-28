#ifndef BASELINER_MANAGER_HPP
#define BASELINER_MANAGER_HPP
#include <baseliner/ConfigFile.hpp>
#include <baseliner/Metadata.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Recipe.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/Components.hpp>
#include <baseliner/managers/GeneralStorage.hpp>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
namespace Baseliner {

  class Manager {
  public:
    static auto instance() -> Manager * {
      static Manager manager;
      return &manager;
    }

    // TODO do same as in preset management ....
    auto build_recipe(const Recipe &recipe, PresetSet &pre_set) -> std::pair<
        std::variant<std::function<std::shared_ptr<IBenchmark>()>, std::function<std::shared_ptr<ISuite>()>>,
        std::function<void()>> {
      auto backend_it = m_backend_presets.find(recipe.m_backend.m_name);
      auto case_it = m_case_presets.find(recipe.m_case.m_name);
      auto benchmark_it = m_benchmark_presets.find(recipe.m_benchmak.m_name);
      auto stopping_it = m_stopping_presets.find(recipe.m_stopping.m_name);
      if (backend_it == m_backend_presets.end()) {
        throw std::runtime_error("Baseliner Error : Backend " + recipe.m_backend.m_name + " not found");
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
      auto backend_preset_it = backend_it->second.find(recipe.m_backend.m_preset);
      auto case_preset_it = case_it->second.find(recipe.m_case.m_preset);
      auto benchmark_preset_it = benchmark_it->second.find(recipe.m_benchmak.m_preset);
      auto stopping_preset_it = stopping_it->second.find(recipe.m_stopping.m_preset);
      auto stat_preset_it = m_stats_presets.find(recipe.m_stats.m_preset);
      if (backend_preset_it == backend_it->second.end()) {
        throw std::runtime_error("Baseliner Error : Preset " + recipe.m_case.m_preset + " not found for Backend " +
                                 recipe.m_case.m_name);
      }
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
      pre_set.insert({recipe.m_case.m_name, case_preset_it->first, case_preset_it->second.m_description,
                      case_preset_it->second.m_options});
      pre_set.insert({recipe.m_benchmak.m_name, benchmark_preset_it->first, benchmark_preset_it->second.m_description,
                      benchmark_preset_it->second.m_options});
      pre_set.insert({recipe.m_stopping.m_name, stopping_preset_it->first, stopping_preset_it->second.m_description,
                      stopping_preset_it->second.m_options});
      pre_set.insert({recipe.m_stats.m_name, stat_preset_it->first, stat_preset_it->second.m_description,
                      stat_preset_it->second.m_options});
      pre_set.insert({recipe.m_backend.m_name, backend_preset_it->first, backend_preset_it->second.m_description,
                      backend_preset_it->second.m_options});
      auto case_preset = case_preset_it->second;
      auto benchmark_preset = benchmark_preset_it->second;
      auto stopping_preset = stopping_preset_it->second;
      auto stat_preset = stat_preset_it->second;
      auto backend_preset = backend_preset_it->second;

      // Impl exists and preset exists
      auto backend_storage_it = m_backends_storage.find(recipe.m_backend.m_name);
      if (backend_storage_it == m_backends_storage.end()) {
        throw std::runtime_error("Baseliner Error : Backend " + recipe.m_backend.m_name +
                                 " not found in Backend Storage");
      }
      auto *backend_storage = backend_storage_it->second;
      auto setup_backend = [backend_storage, backend_preset]() -> void {
        backend_storage->apply_backend_preset(std::get<OptionsMap>(backend_preset.m_options));
      };
      // Adding General stats to general_stats
      // adding notfound_stats aka Device Specific stats to nofound_stats.
      std::vector<std::string> notfound_stats;
      std::vector<std::function<void(std::shared_ptr<Stats::StatsEngine>)>> general_stats;
      for (const auto &stat : std::get<std::vector<std::string>>(stat_preset.m_options)) {
        if (m_stats_storage.has(stat)) {
          general_stats.push_back(m_stats_storage.at(stat));
        } else {
          notfound_stats.push_back(stat);
        }
      }

      auto benchmark_recipe = backend_storage->get_benchmark_with_case(
          recipe.m_benchmak.m_name, std::get<OptionsMap>(benchmark_preset.m_options), recipe.m_case.m_name,
          std::get<OptionsMap>(case_preset.m_options), notfound_stats);

      if (!m_stopping_storage.has(recipe.m_stopping.m_name)) {
        throw std::runtime_error("Baseliner Error : Stopping " + recipe.m_stopping.m_name + " not found in Storage");
      }
      auto stopping_criterion = m_stopping_storage.at(recipe.m_stopping.m_name);
      auto stopping_criterion_w_preset =
          inject_unique_preset<StoppingCriterion>(stopping_criterion, std::get<OptionsMap>(stopping_preset.m_options));

      auto benchmark_with_stopping_and_stats = [benchmark_recipe, stopping_criterion_w_preset,
                                                general_stats]() -> std::shared_ptr<IBenchmark> {
        auto bench = benchmark_recipe();
        bench->set_stopping_criterion(stopping_criterion_w_preset);
        bench->add_stats(general_stats);
        return bench;
      };

      if (recipe.m_suite.has_value()) {
        auto suite = recipe.m_suite.value();
        auto suite_it = m_suite_presets.find(suite.m_name);
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
        pre_set.insert({suite.m_name, suite_preset_it->first, suite_preset_it->second.m_description,
                        suite_preset_it->second.m_options});
        auto suite_creator = m_suite_storage.at(suite.m_name);
        auto suite_w_preset =
            inject_shared_preset(suite_creator, std::get<OptionsMap>(suite_preset_it->second.m_options));
        auto suite_w_everything = [suite_w_preset, benchmark_with_stopping_and_stats]() -> std::shared_ptr<ISuite> {
          auto suite_ptr = suite_w_preset();
          suite_ptr->set_benchmark(benchmark_with_stopping_and_stats);
          return suite_ptr;
        };
        return std::make_pair<
            std::variant<std::function<std::shared_ptr<IBenchmark>()>, std::function<std::shared_ptr<ISuite>()>>,
            std::function<void()>>(suite_w_everything, setup_backend);
      }
      return std::make_pair<
          std::variant<std::function<std::shared_ptr<IBenchmark>()>, std::function<std::shared_ptr<ISuite>()>>,
          std::function<void()>>(benchmark_with_stopping_and_stats, setup_backend);
    }

    /**
     ** list things
     **/

    auto list_cases_presets() const -> std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> {
      return m_case_presets;
    }
    auto list_benchmark_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> {
      return m_benchmark_presets;
    }
    auto list_suite_presets() const -> std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> {
      return m_suite_presets;
    }
    auto list_stopping_presets() const
        -> std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> {
      return m_stopping_presets;
    }
    auto list_general_stats_presets() const -> std::unordered_map<std::string, InnerPreset> {
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
    void register_component(const std::string &name, const ComponentType &type,
                            const std::variant<OptionsMap, std::vector<std::string>> &default_opt) {
      if (m_implementation_to_component.find(name) != m_implementation_to_component.end()) {
        throw std::runtime_error("Baseliner Error: The component name " + name + "  is already taken by a " +
                                 component_to_string(m_implementation_to_component[name]));
      }
      m_implementation_to_component[name] = type;
      add_preset(name, std::string(DEFAULT_PRESET), InnerPreset{std::string(DEFAULT_DESCRIPTION), default_opt}, type);
    }

    void register_backend(const std::string &name, IBackendStorage *backend,
                          const std::variant<OptionsMap, std::vector<std::string>> &default_op) {
      if (m_backends_storage.find(name) != m_backends_storage.end()) {
        throw std::runtime_error("Baseliner Error : Two backends with the same name registered : " + name);
      }
      m_backends_storage[name] = backend;
      register_component(name, ComponentType::BACKEND, default_op);
    }
    void register_general_stat(const std::string &name,
                               const std::function<void(std::shared_ptr<Stats::StatsEngine>)> &stat_factory) {
      m_stats_storage.insert(name, stat_factory);
    }
    void register_suite(const std::string &name, const std::function<std::shared_ptr<ISuite>()> &suite_factory) {
      m_suite_storage.insert(name, suite_factory);
      register_component(name, ComponentType::SUITE, suite_factory()->gather_options());
    }
    void register_stopping(const std::string &name,
                           const std::function<std::unique_ptr<StoppingCriterion>()> &stopping_factory) {
      m_stopping_storage.insert(name, stopping_factory);
      register_component(name, ComponentType::STOPPING, stopping_factory()->gather_options());
    }

    auto get_preset_definitions(ComponentType type) const -> std::vector<PresetDefinition> {
      const std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> *map = nullptr;
      const std::unordered_map<std::string, InnerPreset> *map_stat = nullptr;
      std::vector<PresetDefinition> defs;
      defs.reserve(presets_size(type));
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
      case BACKEND:
        map = &m_backend_presets;
        break;
      case STAT:
        map_stat = &m_stats_presets;
        break;
      }
      if (type == ComponentType::STAT) {
        for (const auto &[name, inner_preset] : (*map_stat)) {
          defs.push_back(
              PresetDefinition{component_to_string(type), name, inner_preset.m_description, inner_preset.m_options});
        }
      } else {
        for (const auto &[interface_name, inner_map] : (*map)) {
          for (const auto &[name, inner_preset] : inner_map) {
            defs.push_back(PresetDefinition{interface_name, name, inner_preset.m_description, inner_preset.m_options});
          }
        }
      }
      return defs;
    }
    auto all_presets_size() const -> size_t {
      size_t total_size = 0;
      total_size += presets_size(ComponentType::BENCHMARK);
      total_size += presets_size(ComponentType::CASE);
      total_size += presets_size(ComponentType::STAT);
      total_size += presets_size(ComponentType::STOPPING);
      total_size += presets_size(ComponentType::SUITE);
      return total_size;
    }
    auto presets_size(ComponentType type) const -> size_t {
      const std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> *map = nullptr;
      const std::unordered_map<std::string, InnerPreset> *map_stat = nullptr;
      size_t preset_size = 0;
      switch (type) {
      case NONE: {
        throw std::runtime_error("presets_size called with NONE ComponentType");
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
      case BACKEND:
        map = &m_backend_presets;
        break;
      case STAT:
        map_stat = &m_stats_presets;
        break;
      }

      if (type == ComponentType::STAT) {
        preset_size = map_stat->size();
      } else {
        for (const auto &[name, inner_map] : (*map)) {
          preset_size += inner_map.size();
        }
      }

      return preset_size;
    }

    /**
     ** Registring Preset
     **/

    auto generate_metadata() const -> Metadata {
      Metadata metadata{};
      for (const auto &[interface_name, map] : m_benchmark_presets) {
        Ingredient ingredient{interface_name, {}};
        for (const auto &[preset_name, preset] : map) {
          ingredient.m_presets.push_back({preset_name, preset});
        }
        metadata.m_benchmarks.push_back(ingredient);
      }
      for (const auto &[interface_name, map] : m_case_presets) {
        Ingredient ingredient{interface_name, {}};
        for (const auto &[preset_name, preset] : map) {
          ingredient.m_presets.push_back({preset_name, preset});
        }
        metadata.m_cases.push_back(ingredient);
      }
      for (const auto &[interface_name, map] : m_stopping_presets) {
        Ingredient ingredient{interface_name, {}};
        for (const auto &[preset_name, preset] : map) {
          ingredient.m_presets.push_back({preset_name, preset});
        }
        metadata.m_stopping_criterions.push_back(ingredient);
      }
      for (const auto &[interface_name, map] : m_suite_presets) {
        Ingredient ingredient{interface_name, {}};
        for (const auto &[preset_name, preset] : map) {
          ingredient.m_presets.push_back({preset_name, preset});
        }
        metadata.m_suites.push_back(ingredient);
      }
      for (const auto &[preset_name, preset] : m_stats_presets) {
        Preset build_preset{preset_name, preset};
        metadata.m_stats_presets.push_back(build_preset);
      }
      metadata.m_general_stats = m_stats_storage.list();
      for (auto [_, backend] : m_backends_storage) {
        metadata.m_backends.push_back(backend->generate_backend_metadata());
      }
      return metadata;
    };

    auto generate_example_config() const -> Config {
      Config def_config;
      def_config.m_baseliner_version = Version::string;
      std::vector<PresetDefinition> preset_defs;
      preset_defs.reserve(all_presets_size());
      std::vector<PresetDefinition> temp_preset;
      temp_preset = get_preset_definitions(ComponentType::BENCHMARK);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      temp_preset = get_preset_definitions(ComponentType::CASE);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      temp_preset = get_preset_definitions(ComponentType::SUITE);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      temp_preset = get_preset_definitions(ComponentType::STOPPING);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      temp_preset = get_preset_definitions(ComponentType::STAT);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      temp_preset = get_preset_definitions(ComponentType::BACKEND);
      preset_defs.insert(preset_defs.end(), temp_preset.begin(), temp_preset.end());
      def_config.m_presets = preset_defs;
      std::vector<Recipe> recipes;
      Recipe default_recipe;
      default_recipe.m_benchmak = {std::string(DEFAULT_BENCHMARK), std::string(DEFAULT_PRESET)};
      default_recipe.m_stopping = {std::string(DEFAULT_STOPPING), std::string(DEFAULT_PRESET)};
      default_recipe.m_stats = {std::string(component_to_string(ComponentType::STAT)), std::string(DEFAULT_PRESET)};

      for (const auto &[backend_name, backend] : m_backends_storage) {
        for (const auto &[case_name, preset] : m_case_presets) {
          if (backend->has_case(case_name)) {
            Recipe inner = default_recipe;
            inner.m_backend = {backend_name, std::string(DEFAULT_PRESET)};
            inner.m_case = {case_name, std::string(DEFAULT_PRESET)};
            recipes.push_back(inner);
          }
        }
      }
      def_config.m_recipes = recipes;
      return def_config;
    };

    void add_presets(const std::vector<PresetDefinition> &presets_definitions) {
      for (const auto &preset_def : presets_definitions) {
        ComponentType type = ComponentType::NONE;
        if (m_implementation_to_component.find(preset_def.m_implementation_name) !=
            m_implementation_to_component.end()) {
          type = m_implementation_to_component.at(preset_def.m_implementation_name);
        }

        if (type == ComponentType::NONE) {
          throw std::invalid_argument("Baseliner Error: Couldn't find Component type for implementation : " +
                                      preset_def.m_implementation_name);
        }

        InnerPreset inner{preset_def.m_description, preset_def.m_options};
        try {
          add_preset(preset_def.m_implementation_name, preset_def.m_preset_name, inner, type);
        } catch (const std::exception &e) {
          std::cerr << "Baseliner Error: " << e.what() << '\n';
          std::cerr << "Keeping original preset\n";
        }
      }

      // Recipes processing deferred as requested.
    };

  private:
    void add_preset(const std::string &name, const std::string &preset_name, const InnerPreset &preset,
                    ComponentType type) {
      std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> *map = nullptr;
      std::unordered_map<std::string, InnerPreset> *map_stat = nullptr;
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
      case STAT:
        map_stat = &m_stats_presets;
        break;
      case BACKEND:
        map = &m_backend_presets;
        break;
      }

      std::unordered_map<std::string, InnerPreset> *specific;
      if (type == ComponentType::STAT) {
        specific = map_stat;
      } else {
        specific = &(*map)[name];
      }

      if (specific->find(preset_name) != specific->end() && preset_name != std::string(DEFAULT_PRESET)) {
        std::cerr << "Baseliner Warning : Preset " << preset_name << " overrides already defined preset\n";
      }
      (*specific)[preset_name] = preset;
    }

    std::unordered_map<std::string, IBackendStorage *> m_backends_storage;
    GeneralStatsStorage m_stats_storage;
    SuiteStorage m_suite_storage;
    StoppingCriterionStorage m_stopping_storage;
    std::unordered_map<std::string, ComponentType> m_implementation_to_component;
    // Preset Management item with IOptions
    std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> m_case_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> m_benchmark_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> m_suite_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> m_stopping_presets;
    std::unordered_map<std::string, std::unordered_map<std::string, InnerPreset>> m_backend_presets;

    // Preset for stats (just names)
    std::unordered_map<std::string, InnerPreset> m_stats_presets;
    // Defaults
    std::string m_default_benchmarl = std::string(DEFAULT_BENCHMARK);
    std::string m_default_stopping = std::string(DEFAULT_STOPPING);
  };
} // namespace Baseliner
#endif // BASELINER_MANAGER_HPP