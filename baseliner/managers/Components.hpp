#ifndef BASELINER_COMPONENT_HPP
#define BASELINER_COMPONENT_HPP
#include <sstream>
#include <string>

constexpr std::string_view DEFAULT_BENCHMARK = "Benchmark";
constexpr std::string_view DEFAULT_STOPPING = "StoppingCriterion";
constexpr std::string_view DEFAULT_PRESET = "default";
constexpr std::string_view DEFAULT_DESCRIPTION = "Default preset";

enum ComponentType : uint8_t {
  NONE,
  CASE,
  BENCHMARK,
  SUITE,
  STAT,
  STOPPING,
  BACKEND
};

static auto component_to_string(const ComponentType &type) -> std::string {
  switch (type) {
  case NONE:
    return "";
  case CASE:
    return "Case";
  case BENCHMARK:
    return "Benchmark";
  case SUITE:
    return "Suite";
  case STOPPING:
    return "StoppingCriterion";
  case STAT:
    return "Stat";
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
  if (str == "Suite") {
    return SUITE;
  }
  if (str == "StoppingCriterion") {
    return STOPPING;
  }
  if (str == "Stat") {
    return STAT;
  }
  return NONE;
}
#endif // BASELINER_COMPONENT_HPP
