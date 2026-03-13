#ifndef BASELINER_COMPONENT_HPP
#define BASELINER_COMPONENT_HPP
#include <baseliner/Options.hpp>
#include <sstream>
#include <string>
namespace Baseliner {

  constexpr std::string_view DEFAULT_PRESET = "default";
  constexpr std::string_view DEFAULT_DESCRIPTION = "Default preset";
  constexpr std::string_view DEFAULT_STAT = DEFAULT_PRESET;

  enum ComponentType : uint8_t {
    NONE,
    CASE,
    BENCHMARK,
    STOPPING,
    BACKEND
  };

  struct ComponentPreset {
    std::string m_description;
    OptionsMap m_options;
  };

  struct StatsPreset {
    std::string m_description;
    std::vector<std::string> m_stat_names;
    OptionsMap m_stat_options;
  };

  static auto component_to_string(const ComponentType &type) -> std::string;
  static auto string_to_component(const std::string_view &str) -> ComponentType;

} // namespace Baseliner
#endif // BASELINER_COMPONENT_HPP