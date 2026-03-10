#ifndef BASELINER_STORAGE_MANAGER_HPP
#define BASELINER_STORAGE_MANAGER_HPP
#include <baseliner/Protocol.hpp>
#include <baseliner/managers/BackendStorage.hpp>
#include <baseliner/managers/GeneralStorage.hpp>
#include <iostream>
#include <unordered_map>
namespace Baseliner {
  class StorageManager {
  public:
    static auto instance() -> StorageManager * {
      static StorageManager storage{};
      return &storage;
    }

    /*
     * Registring stuff
     *
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