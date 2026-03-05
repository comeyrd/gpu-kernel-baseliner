#pragma once
#include <string>
#include <string_view>
namespace Baseliner::Version {
  inline constexpr int major = 0; // NOLINT
  inline constexpr int minor = 9; // NOLINT
  inline constexpr int patch = 0; // NOLINT
  inline constexpr std::string_view string_view = "0.9.0";
  inline auto string() -> std::string {
    return std::string(string_view);
  }
} // namespace Baseliner::Version