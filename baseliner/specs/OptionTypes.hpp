#ifndef BASELINER_CORE_OPTIONTYPES_HPP
#define BASELINER_CORE_OPTIONTYPES_HPP
#include <optional>
#include <string>
#include <unordered_map>
#ifdef BASELINER_FULL_LIBRARY
#include <baseliner/cli/Serializer.hpp>
#endif
namespace Baseliner {
  struct Option {
    std::optional<std::string> description;
    std::string value;
  };
#ifdef BASELINER_FULL_LIBRARY
  DESCRIBE(Option, FIELD(description), FIELD(value))
#endif

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;
} // namespace Baseliner

#endif // BASELINER_OPTION_TYPES