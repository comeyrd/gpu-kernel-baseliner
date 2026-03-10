#include <baseliner/managers/StorageManager.hpp>

namespace Baseliner {
  void StorageManager::register_component(const std::string &name, const ComponentType &type,
                                          const OptionsMap &default_opt) {
    if (m_components.find(name) != m_components.end()) {
      throw std::runtime_error("Baseliner Error: The component name " + name + "  is already taken by a " +
                               component_to_string(m_components[name]));
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
      throw std::runtime_error("Baseliner Error : Two backends with the same name registered : " + name);
    }
    m_backends_storage[name] = storage;
    register_component(name, ComponentType::BACKEND, options);
  }
  void StorageManager::register_stopping(const std::string &name, const StoppingCriterionFactory &stopping_factory) {
    m_stopping_storage.insert(name, stopping_factory);
    register_component(name, ComponentType::STOPPING, stopping_factory()->gather_options());
  }
  void StorageManager::register_general_stat(const std::string &name, const StatsFactory &stat_factory) {
    m_stats_storage.insert(name, stat_factory);
  }

} // namespace Baseliner