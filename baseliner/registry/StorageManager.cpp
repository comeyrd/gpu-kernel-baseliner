#include <baseliner/core/Error.hpp>
#include <baseliner/registry/Components.hpp>
#include <baseliner/registry/Factories.hpp>
#include <baseliner/registry/StorageManager.hpp>

namespace Baseliner {
  /*
   * Registring components, stats, presets ...
   */
  void StorageManager::register_component(const std::string &name, const ComponentType &type,
                                          const OptionsMap &default_opt) {
    if (m_components.find(name) != m_components.end()) {
      throw Errors::component_already_exists(name, component_to_string(m_components[name]));
    }
    m_components[name] = type;
    add_component_preset(name, std::string(DEFAULT_PRESET),
                         ComponentPreset{std::string(DEFAULT_DESCRIPTION), default_opt});
  }
  void StorageManager::register_stat_preset(const std::string &name, const StatsPreset &preset) {
    add_stat_preset(name, preset);
  }
  void StorageManager::register_component_preset(const std::string &name, const std::string &preset_name,
                                                 const ComponentPreset &preset) {
    add_component_preset(name, preset_name, preset);
  }
  void StorageManager::register_backend(const std::string &name, IBackendStorage *storage, const OptionsMap &options) {
    if (m_backends_storage.find(name) != m_backends_storage.end()) {
      throw Errors::component_already_exists(name, component_to_string(ComponentType::BACKEND));
    }
    m_backends_storage[name] = storage;
    register_component(name, ComponentType::BACKEND, options);
  }
  void StorageManager::register_stopping(const std::string &name, const StoppingCriterionFactory &stopping_factory) {
    m_stopping_storage.insert(name, stopping_factory);
    register_component(name, ComponentType::STOPPING, stopping_factory()->get_options());
  }
  void StorageManager::register_general_stat(const std::string &name, const StatsFactory &stat_factory) {
    m_stats_storage.insert(name, stat_factory);
  }

  /*
   * Getting factories
   */
  auto StorageManager::get_stopping_criterion_factory(const std::string &name) const -> StoppingCriterionFactory {
    if (m_stopping_storage.has(name)) {
      return m_stopping_storage.at(name);
    }
    throw Errors::not_found(component_to_string(ComponentType::STOPPING), name);
  }
  auto StorageManager::get_benchmark_case_factory(const std::string &backend_name, const std::string &benchmark_name,
                                                  const std::string &case_name) const -> IBenchmarkFactory {
    IBackendStorage *storage = get_backend_storage(backend_name);
    return storage->get_benchmark_with_case(benchmark_name, case_name);
  }
  auto StorageManager::get_stats_factory(const std::string &name) const -> StatsFactory {
    std::optional<StatsFactory> fact = get_stats_factory_noexcept(name);
    if (fact.has_value()) {
      return fact.value();
    }
    throw Errors::stat_not_found(name);
  }

  auto StorageManager::get_backend_stats_factory(const std::string &backend, const std::string &name) const
      -> StatsFactory {
    std::optional<StatsFactory> fact = get_backend_stats_factory_noexcept(backend, name);
    if (fact.has_value()) {
      return fact.value();
    }
    throw Errors::stat_not_found_backend(name, backend);
  };
  auto StorageManager::get_stats_factory_noexcept(const std::string &name) const -> std::optional<StatsFactory> {
    if (m_stats_storage.has(name)) {
      return m_stats_storage.at(name);
    }
    return {};
  }

  auto StorageManager::get_backend_stats_factory_noexcept(const std::string &backend, const std::string &name) const
      -> std::optional<StatsFactory> {
    IBackendStorage *storage = get_backend_storage(backend);
    if (storage->has_stat(name)) {
      return m_stats_storage.at(name);
    }
    return {};
  };
  [[nodiscard]] auto StorageManager::get_combined_stats_factories(const std::string &backend,
                                                                  const std::vector<std::string> &stat_names) const
      -> StatsFactory {
    std::vector<StatsFactory> stat_factories;
    stat_factories.reserve(stat_names.size());
    for (const auto &name : stat_names) {
      std::optional<StatsFactory> fact = get_stats_factory_noexcept(name);
      if (fact.has_value()) {
        stat_factories.push_back(fact.value());
      } else {
        std::optional<StatsFactory> backend_fact = get_backend_stats_factory_noexcept(backend, name);
        if (backend_fact.has_value()) {
          stat_factories.push_back(backend_fact.value());
        } else {
          throw Errors::stat_not_found_either_general_or_backend(name, backend);
        }
      }
    }
    return [stat_factories](std::shared_ptr<Stats::StatsEngine> engine) -> void {
      for (const auto &fact : stat_factories) {
        fact(engine);
      }
    };
  };

  auto StorageManager::get_backend_storage(const std::string &name) const -> IBackendStorage * {
    if (m_backends_storage.find(name) != m_backends_storage.end()) {
      return m_backends_storage.at(name);
    }
    throw Errors::not_found(component_to_string(ComponentType::BACKEND), name);
  };

  auto StorageManager::get_component_preset(const std::string &impl, const std::string &preset) const
      -> ComponentPreset {
    check_component_preset(impl, preset);
    return m_component_presets.at(impl).at(preset);
  };

  auto StorageManager::get_stats_preset(const std::string &name) const -> StatsPreset {
    if (m_stats_presets.find(name) != m_stats_presets.end()) {
      return m_stats_presets.at(name);
    }
    throw Errors::stat_not_found(name);
  };

  auto StorageManager::list_components() const -> ComponentList {
    return {m_components.begin(), m_components.end()};
  };
  auto StorageManager::list_components(ComponentType wanted_type) -> std::vector<std::string> {
    std::vector<std::string> impl;
    impl.reserve(m_components.size());
    for (const auto &[name, type] : m_components) {
      if (type == wanted_type) {
        impl.push_back(name);
      }
    }
    impl.shrink_to_fit();
    return impl;
  }

  auto StorageManager::list_stats() const -> std::vector<std::string> {
    std::vector<std::string> keys;
    keys.reserve(m_stats_presets.size());
    for (const auto &[key, _] : m_stats_presets) {
      keys.push_back(key);
    }
    return keys;
  };
  auto StorageManager::list_backends() const -> std::vector<std::string> {
    std::vector<std::string> keys;
    keys.reserve(m_backends_storage.size());
    for (const auto &[key, _] : m_backends_storage) {
      keys.push_back(key);
    }
    return keys;
  };
  auto StorageManager::list_backend_components(std::string &backend) const -> ComponentList {
    IBackendStorage *storage = get_backend_storage(backend);
    return storage->list_components();
  };
  auto StorageManager::list_backend_stats(std::string &backend) const -> std::vector<std::string> {
    IBackendStorage *storage = get_backend_storage(backend);
    return storage->list_device_stats();
  };

  auto StorageManager::list_component_presets(const std::string &component_name) const -> ComponentPresetList {
    check_component(component_name);
    return {m_component_presets.at(component_name).begin(), m_component_presets.at(component_name).end()};
  };
  auto StorageManager::list_stat_presets() const -> StatsPresetList {
    return {m_stats_presets.begin(), m_stats_presets.end()};
  };
  void StorageManager::check_component(const std::string &name) const {
    if (m_components.find(name) == m_components.end()) {
      throw Errors::not_found("Component", name);
    }
    if (m_component_presets.find(name) == m_component_presets.end()) {
      throw Errors::not_found_in(component_to_string(m_components.at(name)), name, "Presets");
    }
  };
  void StorageManager::check_component_preset(const std::string &component, const std::string &preset) const {
    check_component(component);
    const auto &impl_preset = m_component_presets.at(component);
    if (impl_preset.find(preset) == impl_preset.end()) {
      throw Errors::not_found_in("Preset", preset, component);
    }
  }

  auto StorageManager::get_all_component_presets()
      -> std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> {
    return m_component_presets;
  }
  auto StorageManager::get_all_stats_presets() -> std::unordered_map<std::string, StatsPreset> {
    return m_stats_presets;
  }

} // namespace Baseliner