#ifndef BASELINER_CONVERSION_HPP
#define BASELINER_CONVERSION_HPP
#include <string>
namespace Baseliner {
  namespace Conversion {
    template <typename T>
    T baseliner_from_string(const std::string &val);
    template <typename T>
    std::string baseliner_to_string(const T &val);

    inline std::string bool_to_string(bool value) {
      return std::to_string(static_cast<int>(value));
    };
    inline bool string_to_bool(std::string value) {
      int i_value = std::stoi(value);
      return (i_value != 0);
    };
    // To string
    template <>
    inline std::string baseliner_to_string<bool>(const bool &val) {
      return bool_to_string(val);
    };
    template <>
    inline std::string baseliner_to_string<int>(const int &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<long>(const long &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<long long>(const long long &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<unsigned int>(const unsigned int &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<unsigned long>(const unsigned long &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<unsigned long long>(const unsigned long long &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<float>(const float &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<double>(const double &val) {
      return std::to_string(val);
    }
    template <>
    inline std::string baseliner_to_string<long double>(const long double &val) {
      return std::to_string(val);
    }
    // From String
    template <>
    inline bool baseliner_from_string(const std::string &val) {
      return string_to_bool(val);
    };
    template <>
    inline int baseliner_from_string<int>(const std::string &val) {
      return std::stoi(val);
    }
    template <>
    inline long baseliner_from_string<long>(const std::string &val) {
      return std::stol(val);
    }
    template <>
    inline long long baseliner_from_string<long long>(const std::string &val) {
      return std::stoll(val);
    }
    template <>
    inline unsigned int baseliner_from_string<unsigned int>(const std::string &val) {
      return static_cast<unsigned int>(std::stoul(val));
    }
    template <>
    inline unsigned long baseliner_from_string<unsigned long>(const std::string &val) {
      return std::stoul(val);
    }
    template <>
    inline unsigned long long baseliner_from_string<unsigned long long>(const std::string &val) {
      return std::stoull(val);
    }
    template <>
    inline float baseliner_from_string<float>(const std::string &val) {
      return std::stof(val);
    }
    template <>
    inline double baseliner_from_string<double>(const std::string &val) {
      return std::stod(val);
    }
    template <>
    inline long double baseliner_from_string<long double>(const std::string &val) {
      return std::stold(val);
    }
  } // namespace Conversion

} // namespace Baseliner
#endif // BASELINER_CONVERSION_HPP
