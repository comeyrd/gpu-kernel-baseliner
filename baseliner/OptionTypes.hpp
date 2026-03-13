#ifndef BASELINER_OPTION_TYPES
#define BASELINER_OPTION_TYPES
#include <string>
#include <unordered_map>
namespace {
  struct Option {
    std::string m_description;
    std::string m_value;
  };

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;
} // namespace

#endif // BASELINER_OPTION_TYPES