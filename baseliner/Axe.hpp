#ifndef BASELINER_AXE_HPP
#define BASELINER_AXE_HPP
#include <baseliner/Options.hpp>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {

  struct Axe {
    std::string axe_name;
    std::string axe_description;
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<std::string> m_values;
  };
} // namespace Baseliner
#endif // BASELINER_AXE_HPP