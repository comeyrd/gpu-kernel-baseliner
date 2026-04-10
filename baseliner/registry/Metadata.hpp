#ifndef BASELINER_REGISTRY_METADATA_HPP
#define BASELINER_REGISTRY_METADATA_HPP
#include <baseliner/registry/Components.hpp>
namespace Baseliner {

  struct Metadata {
    std::unordered_map<std::string, std::vector<std::string>> components;
    std::vector<std::string> stats;
    std::unordered_map<std::string, std::unordered_map<std::string, ComponentPreset>> component_presets;
    std::unordered_map<std::string, StatsPreset> stats_presets;
    std::unordered_map<std::string, OptionsMap> stats_options;
    std::unordered_map<std::string, std::vector<std::string>> hardware_stats;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> hardware_components;
    std::unordered_map<std::string, std::unordered_map<std::string, OptionsMap>> hardware_stat_options;
  };
  DESCRIBE(Metadata, FIELD(components), FIELD(stats), FIELD(component_presets), FIELD(stats_presets),
           FIELD(stats_options), FIELD(hardware_stats), FIELD(hardware_components), FIELD(hardware_stat_options))

}; // namespace Baseliner
#endif // BASELINER_REGISTRY_METADATA_HPP
