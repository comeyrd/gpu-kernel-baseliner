#ifndef BASELINER_STORAGE_MANAGER_HPP
#define BASELINER_STORAGE_MANAGER_HPP
#include "baseliner/managers/Components.hpp"
#include "baseliner/managers/Factories.hpp"
#include <baseliner/Protocol.hpp>
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/GeneralStorage.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace Baseliner {

  class StorageManager {
  public:
    static auto instance() -> StorageManager * {
      static StorageManager storage{};
      return &storage;
    }

    /*
     * Registring components, stats, presets ...
     */
    void register_component(const std::string &name, const ComponentType &type, const OptionsMap &default_opt);
    void register_stat_preset(const std::string &name, const StatsPreset &preset);
    void register_component_preset(const std::string &name, const std::string &preset_name,
                                   const ComponentPreset &preset);
    void register_backend(const std::string &name, IBackendStorage *storage, const OptionsMap &options);
    void register_stopping(const std::string &name, const StoppingCriterionFactory &stopping_factory);
    // TODO maybe let the stat thiny only to the engine ?
    // Maybe create something else than the manager ?
    void register_general_stat(const std::string &name, const StatsFactory &stat_factory);

    void load_protocol_presets(const Protocol &protocol);

    /*
     * Getting factories
     */
    [[nodiscard]] auto get_stopping_criterion_factory(const std::string &name) const -> StoppingCriterionFactory;
    [[nodiscard]] auto get_benchmark_case_factory(const std::string &backend_name, const std::string &benchmark_name,
                                                  const std::string &case_name) const -> IBenchmarkFactory;
    [[nodiscard]] auto get_stats_factory(const std::string &name) const -> StatsFactory;
    [[nodiscard]] auto get_backend_stats_factory(const std::string &backend, const std::string &name) const
        -> StatsFactory;
    [[nodiscard]] auto get_stats_factory_noexcept(const std::string &name) const -> std::optional<StatsFactory>;
    [[nodiscard]] auto get_backend_stats_factory_noexcept(const std::string &backend, const std::string &name) const
        -> std::optional<StatsFactory>;
    [[nodiscard]] auto get_combined_stats_factories(const std::string &backend,
                                                    const std::vector<std::string> &stat_names) const -> StatsFactory;
    /*
     * Getting preset
     */
    [[nodiscard]] auto get_component_preset(const std::string &impl, const std::string &preset) const
        -> ComponentPreset;
    [[nodiscard]] auto get_stats_preset(const std::string &name) const -> StatsPreset;

    /*
     * Listing in storage
     */
    [[nodiscard]] auto list_components() const -> ComponentList;
    [[nodiscard]] auto list_stats() const -> std::vector<std::string>;
    [[nodiscard]] auto list_backends() const -> std::vector<std::string>;
    [[nodiscard]] auto list_backend_components(std::string &backend) const -> ComponentList;
    [[nodiscard]] auto list_backend_stats(std::string &backend) const -> std::vector<std::string>;

    /*
     * Listing presets
     */
    [[nodiscard]] auto list_component_presets(const std::string &component_name) const -> ComponentPresetList;
    [[nodiscard]] auto list_stat_presets() const -> StatsPresetList;

  private:
    /*
     * Presets
     */

    void add_component_preset(const std::string &name, const std::string &preset_name, const ComponentPreset &preset) {
      if (m_component_presets.find(name) == m_component_presets.end()) {
        m_component_presets[name] = {};
      }
      if (m_component_presets[name].find(preset_name) != m_component_presets[name].end() &&
          preset_name != std::string(DEFAULT_PRESET)) {
        std::cerr << "Baseliner Warning : Preset " << preset_name << " overrides already defined preset\n";
      }
      m_component_presets[name][preset_name] = preset;
    }
    void add_stat_preset(const std::string &name, const StatsPreset &preset) {
      if (m_component_presets.find(name) != m_component_presets.end() && name != std::string(DEFAULT_PRESET)) {
        std::cerr << "Baseliner Warning : Preset " << name << " overrides already defined preset\n";
      }
      m_stats_presets[name] = preset;
    }

    [[nodiscard]] auto get_backend_storage(const std::string &name) const -> IBackendStorage *;
    void check_component(const std::string &name) const;
    void check_component_preset(const std::string &component, const std::string &preset) const;
    StorageManager() = default;
    // Storage
    GeneralStatsStorage m_stats_storage;
    StoppingCriterionStorage m_stopping_storage;
    std::unordered_map<std::string, IBackendStorage *> m_backends_storage;
    // Stats
    std::unordered_map<std::string, StatsPreset> m_stats_presets;
    // Components
    std::unordered_map<std::string, ComponentType> m_components;
    std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> m_component_presets;
  };

} // namespace Baseliner

#endif // BASELINER_STORAGE_MANAGER_HPP