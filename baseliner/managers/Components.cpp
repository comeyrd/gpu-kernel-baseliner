#include <baseliner/managers/Components.hpp>
#include <string>
namespace Baseliner {
  static auto component_to_string(const ComponentType &type) -> std::string {
    switch (type) {
    case NONE:
      return "";
    case CASE:
      return "Case";
    case BENCHMARK:
      return "Benchmark";
    case STOPPING:
      return "StoppingCriterion";
    default:
      return "";
    }
  }
  static auto string_to_component(const std::string_view &str) -> ComponentType {
    if (str == "Case") {
      return CASE;
    }
    if (str == "Benchmark") {
      return BENCHMARK;
    }
    if (str == "StoppingCriterion") {
      return STOPPING;
    }
    return NONE;
  }
} // namespace Baseliner