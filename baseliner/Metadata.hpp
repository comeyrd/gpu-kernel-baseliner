#ifndef BASELINER_METADATA_HPP
#define BASELINER_METADATA_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Version.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  struct InnerOptionPreset {
    std::string m_description;
    OptionsMap m_map;
  };
  struct OptionPreset {
    std::string m_name;
    InnerOptionPreset m_preset;
  };
  struct InnerStatPreset {
    std::string m_description;
    std::vector<std::string> m_stats;
  };

  struct BackendMetadata {
    std::string m_name;
    std::vector<std::string> m_benchmaks;
    std::vector<std::string> m_cases;
    std::vector<std::string> m_stats;
  };
  struct Ingredient {
    std::string m_name;
    std::vector<OptionPreset> m_presets;
  };

  struct StatPreset {
    std::string m_name;
    InnerStatPreset m_preset;
  };

  struct Metadata {
    std::string baseliner_version = std::string(Version::string);
    std::vector<Ingredient> m_benchmarks;
    std::vector<Ingredient> m_cases;
    std::vector<Ingredient> m_stopping_criterions;
    std::vector<Ingredient> m_suites;
    std::vector<std::string> m_general_stats;
    std::vector<StatPreset> m_stats_presets;
    std::vector<BackendMetadata> m_backends;
  };

} // namespace Baseliner
#endif // BASELINER_METADATA_HPP