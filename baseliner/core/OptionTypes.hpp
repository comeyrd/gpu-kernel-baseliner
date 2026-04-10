#ifndef BASELINER_CORE_OPTIONTYPES_HPP
#define BASELINER_CORE_OPTIONTYPES_HPP
#include <baseliner/cli/Serializer.hpp>
#include <string>
#include <unordered_map>
namespace Baseliner {
  struct Option {
    std::optional<std::string> m_description;
    std::string m_value;
  };
  DESCRIBE(Option, FIELD(m_description), FIELD(m_value))

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;
} // namespace Baseliner

#endif // BASELINER_OPTION_TYPES