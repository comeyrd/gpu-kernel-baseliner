#ifndef BASELINER_CONFIG_FILE_HPP
#define BASELINER_CONFIG_FILE_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Recipe.hpp>
#include <string>
#include <unordered_set>
#include <vector>
namespace Baseliner {

  struct PresetDefinition {
    std::string m_implementation_name;
    std::string m_preset_name;
    std::string m_description;
    std::variant<OptionsMap, std::vector<std::string>> m_options;
  };
  struct PresetEquality {
    bool operator()(const PresetDefinition &lhs, const PresetDefinition &rhs) const {
      return lhs.m_implementation_name == rhs.m_implementation_name && lhs.m_preset_name == rhs.m_preset_name;
    }
  };
  struct PresetHasher {
    std::size_t operator()(const PresetDefinition &p) const {
      std::size_t h1s = std::hash<std::string>{}(p.m_implementation_name);
      std::size_t h2s = std::hash<std::string>{}(p.m_preset_name);
      return h1s ^ (h2s * 31);
    }
  };
  using PresetSet = std::unordered_set<PresetDefinition, PresetHasher, PresetEquality>;

  struct Config {
    std::string m_baseliner_version;
    std::vector<PresetDefinition> m_presets;
    std::vector<Recipe> m_recipes;
  };

} // namespace Baseliner
#endif // BASELINER_CONFIG_FILE_HPP