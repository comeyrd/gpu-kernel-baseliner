#ifndef BASELINER_AXE_HPP
#define BASELINER_AXE_HPP
#include <string>
#include <vector>
namespace Baseliner {

  struct Axe {
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<std::string> m_values;
  };
} // namespace Baseliner
#endif // BASELINER_AXE_HPP