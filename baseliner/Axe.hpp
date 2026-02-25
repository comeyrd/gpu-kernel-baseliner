#ifndef BASELINER_AXE_HPP
#define BASELINER_AXE_HPP
#include <baseliner/Options.hpp>
#include <string>
#include <vector>
namespace Baseliner {

  constexpr std::string_view DEFAULT_SINGLE_AXE_INTERFACE_NAME = "Case";
  constexpr std::string_view DEFAULT_SINGLE_AXE_OPTION_NAME = "work_size";
  constexpr std::string_view DEFAULT_SINGLE_AXE_VALUES[] = {"1", "2", "3", "4", "5"}; // NOLINT
  class SingleAxe : IOption {
  public:
    void register_options() override {
      this->add_option("SingleAxe", "interface_name", "The name of the interface the option is on", m_interface_name);
      this->add_option("SingleAxe", "option_name", "The name of the option", m_option_name);
      this->add_option("SingleAxe", "values", "The selected values of the option", m_values);
    };
    SingleAxe();
    void set_interface_name(const std::string &interface_name) {
      m_interface_name = interface_name;
    }
    void set_option_name(const std::string &option_name) {
      m_option_name = option_name;
    }
    void set_values(const std::vector<std::string> &values) {
      m_values = values;
    }
    auto get_interface_name() -> std::string {
      return m_interface_name;
    };
    auto get_option_name() -> std::string {
      return m_option_name;
    };
    auto get_values() -> std::vector<std::string> {
      return m_values;
    };

  private:
    std::string m_interface_name{DEFAULT_SINGLE_AXE_INTERFACE_NAME};
    std::string m_option_name{DEFAULT_SINGLE_AXE_OPTION_NAME};
    std::vector<std::string> m_values{std::begin(DEFAULT_SINGLE_AXE_VALUES), std::end(DEFAULT_SINGLE_AXE_VALUES)};
  };
} // namespace Baseliner
#endif // BASELINER_AXE_HPP