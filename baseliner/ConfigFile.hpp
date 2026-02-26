#ifndef BASELINER_CONFIG_FILE_HPP
#define BASELINER_CONFIG_FILE_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Recipe.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  struct PresetDefinition {
    std::string m_implementation_name;
    std::string m_preset_name;
    std::string m_description;
    std::variant<OptionsMap, std::vector<std::string>> m_patch;
  };

  struct Config {
    std::string m_baseliner_version;
    std::vector<PresetDefinition> m_presets;
    std::vector<Recipe> m_recipes;
  };

} // namespace Baseliner
#endif // BASELINER_CONFIG_FILE_HPP