#ifndef BASELINER_AXE_HPP
#define BASELINER_AXE_HPP
#include <baseliner/Options.hpp>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {

  class Axe : public IOptionConsumer {
  public:
    Axe(std::string interface_name, std::string option_name, std::vector<std::string> values)
        : m_interface_name(std::move(interface_name)),
          m_option_name(std::move(option_name)),
          m_values(std::move(values)) {};

    auto get_interface_name() const -> std::string {
      return m_interface_name;
    }
    auto get_option_name() const -> std::string {
      return m_option_name;
    }
    auto get_values() const -> std::vector<std::string> {
      return m_values;
    }

  protected:
    void register_options() override {
      add_option("Axe", "interface_name", "The interface name", m_interface_name);
      add_option("Axe", "option_name", "The option name", m_option_name);
      add_option("Axe", "values", "the possible values", m_values);
    }

  private:
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<std::string> m_values;
  };
} // namespace Baseliner
#endif // BASELINER_AXE_HPP