#ifndef BASELINER_AXE_HPP
#define BASELINER_AXE_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Result.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  struct AxeValue {
    std::string m_value;
    AxeValue(std::string name)
        : m_value(name) {};
    AxeValue(const char *name)
        : m_value(name) {};
  };

  struct Axe {
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<AxeValue> m_values;
  };
} // namespace Baseliner
#endif // BASELINER_AXE_HPP