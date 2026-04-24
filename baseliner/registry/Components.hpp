#ifndef BASELINER_REGISTRY_COMPONENTS_HPP
#define BASELINER_REGISTRY_COMPONENTS_HPP
#include <baseliner/cli/Serializer.hpp>
#include <baseliner/core/Options.hpp>
#include <string>
namespace Baseliner {

  constexpr std::string_view DEFAULT_PRESET = "default";
  constexpr std::string_view DEFAULT_DESCRIPTION = "Default preset";
  constexpr std::string_view DEFAULT_STAT = DEFAULT_PRESET;

  enum ComponentType : uint8_t {
    NONE,
    WORKLOAD,
    BENCHMARK,
    STOPPING,
    BACKEND
  };

  struct ComponentPreset {
    std::optional<std::string> description;
    OptionsMap options;
  };
  DESCRIBE(ComponentPreset, FIELD(description), FIELD(options))
  struct StatsPreset {
    std::optional<std::string> description;
    std::vector<std::string> stat_names;
    OptionsMap stat_options;
  };
  DESCRIBE(StatsPreset, FIELD(description), FIELD(stat_names), FIELD(stat_options))

  auto component_to_string(const ComponentType &type) -> std::string;
  auto string_to_component(const std::string_view &str) -> ComponentType;

} // namespace Baseliner
#endif // BASELINER_COMPONENT_HPP