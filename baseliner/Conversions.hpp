#ifndef BASELINER_CONVERSION_HPP
#define BASELINER_CONVERSION_HPP
#include <cstddef>
#include <sstream>
#include <string>
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
  inline auto baseliner_to_string<std::vector<std::string>>(const std::vector<std::string> &val) -> std::string {
    if (val.empty()) {
      return "[]";
    }
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < val.size(); ++i) {
      oss << "\"" << val[i] << "\"" << (i == val.size() - 1 ? "" : ", ");
    }
    oss << "]";
    return oss.str();
  }

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
  inline auto baseliner_from_string<std::vector<std::string>>(const std::string &val) -> std::vector<std::string> {
    std::vector<std::string> result;
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
      result.push_back(item);
    }
    return result;
  }
} // namespace Baseliner::Conversion

#endif // BASELINER_CONVERSION_HPP
