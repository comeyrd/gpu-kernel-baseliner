#ifndef BASELINER_CONVERSION_HPP
#define BASELINER_CONVERSION_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/stats/StatsType.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
namespace Baseliner::Conversion {
  template <typename T>
  auto baseliner_from_string(const std::string &val) -> T;
  template <typename T>
  auto baseliner_to_string(const T &val) -> std::string;

  inline auto bool_to_string(bool value) -> std::string {
    return std::to_string(static_cast<int>(value));
  };
  inline auto string_to_bool(const std::string &value) -> bool {
    const int i_value = std::stoi(value);
    return (i_value != 0);
  };
  template <typename T>
  auto baseliner_vector_from_string(const std::string &val) -> std::vector<T> {
    std::vector<T> result;
    std::string string_v = val;
    if (string_v.front() == '[') {
      string_v.erase(0, 1);
    }
    if (string_v.back() == ']') {
      string_v.pop_back();
    }
    std::stringstream sstream(string_v);
    std::string item;
    while (std::getline(sstream, item, ',')) {
      // Trim whitespace/quotes if necessary
      result.push_back(baseliner_from_string<T>(item));
    }
    return result;
  }
  template <typename T>
  inline auto baseliner_to_string(const std::vector<T> &val) -> std::string {
    if (val.empty()) {
      return "[]";
    }
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < val.size(); ++i) {
      oss << baseliner_to_string<T>(val[i]) << (i == val.size() - 1 ? "" : ", ");
    }
    oss << "]";
    return oss.str();
  }

  // To string
  template <>
  inline auto baseliner_to_string<bool>(const bool &val) -> std::string {
    return bool_to_string(val);
  };
  template <>
  inline auto baseliner_to_string<int>(const int &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<long>(const long &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<long long>(const long long &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<unsigned int>(const unsigned int &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<unsigned long>(const unsigned long &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<unsigned long long>(const unsigned long long &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<float>(const float &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<double>(const double &val) -> std::string {
    return std::to_string(val);
  }
  template <>
  inline auto baseliner_to_string<long double>(const long double &val) -> std::string {
    return std::to_string(val);
  }

  template <>
  inline auto baseliner_to_string<std::string>(const std::string &val) -> std::string {
    return val;
  };
  // From String
  template <>
  inline auto baseliner_from_string(const std::string &val) -> bool {
    return string_to_bool(val);
  };
  template <>
  inline auto baseliner_from_string<int>(const std::string &val) -> int {
    return std::stoi(val);
  }
  template <>
  inline auto baseliner_from_string<long>(const std::string &val) -> long {
    return std::stol(val);
  }
  template <>
  inline auto baseliner_from_string<long long>(const std::string &val) -> long long {
    return std::stoll(val);
  }
  template <>
  inline auto baseliner_from_string<unsigned int>(const std::string &val) -> unsigned int {
    return static_cast<unsigned int>(std::stoul(val));
  }
  template <>
  inline auto baseliner_from_string<unsigned long>(const std::string &val) -> unsigned long {
    return std::stoul(val);
  }
  template <>
  inline auto baseliner_from_string<unsigned long long>(const std::string &val) -> unsigned long long {
    return std::stoull(val);
  }
  template <>
  inline auto baseliner_from_string<float>(const std::string &val) -> float {
    return std::stof(val);
  }
  template <>
  inline auto baseliner_from_string<double>(const std::string &val) -> double {
    return std::stod(val);
  }
  template <>
  inline auto baseliner_from_string<long double>(const std::string &val) -> long double {
    return std::stold(val);
  }
  template <>
  inline auto baseliner_from_string<std::string>(const std::string &val) -> std::string {
    return val;
  }
  template <>
  inline auto baseliner_to_string<float_milliseconds>(const float_milliseconds &val) -> std::string {
    return baseliner_to_string(val.count());
  }

  template <typename T>
  inline auto baseliner_to_string(const ConfidenceInterval<T> &val) -> std::string {
    return baseliner_to_string(val.low) + " ," + baseliner_to_string(val.high);
  }

  template <typename... Types>
  inline auto baseliner_to_string(const std::variant<Types...> &val) -> std::string {
    return std::visit(
        [](auto &&arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            return "null";
          } else {
            return baseliner_to_string(arg);
          }
        },
        val);
  }
} // namespace Baseliner::Conversion

#endif // BASELINER_CONVERSION_HPP
