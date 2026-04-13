#ifndef BASELINER_CORE_OPTIONTYPES_HPP
#define BASELINER_CORE_OPTIONTYPES_HPP
#include <baseliner/cli/Serializer.hpp>
#include <string>
#include <unordered_map>
namespace Baseliner {
  struct Option {
    std::optional<std::string> description;
    std::string value;
  };
  DESCRIBE(Option, FIELD(description), FIELD(value))

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;
} // namespace Baseliner

#endif // BASELINER_OPTION_TYPES