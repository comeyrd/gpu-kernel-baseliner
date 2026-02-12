#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  // IOption
  // Apply the options to its own parameters
  void IOption::propagate_options(const OptionsMap &optionsMap) { // TODO OPTIMIZE
    ensure_initialized();
    for (const auto &[interface_name, options] : optionsMap) {
      for (const auto &[option_name, opt] : options) {
        for (auto &binding : m_options_bindings) {
          if (binding->get_name() == option_name && binding->get_interface_name() == interface_name) {
            try {
              binding->update_value(opt.m_value);
            } catch (std::invalid_argument const &e) {
              std::cout << "Error while applying option : " << binding->get_interface_name() << "." << option_name
                        << "=" << opt.m_value << " Using default value : " << binding->get_value() << '\n';
            }
          }
        }
      }
    }
    on_update();
    for (IOption *consumer : m_consumers) {
      consumer->propagate_options(optionsMap);
    }
  }

  // Describe its own options
  void IOption::gather_options(OptionsMap &opts) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      opts[binding->get_interface_name()][binding->get_name()] =
          Option{binding->get_description(), binding->get_value()};
    }
    for (IOption *consumer : m_consumers) {
      consumer->gather_options(opts);
    }
  }

  auto IOption::gather_options() -> OptionsMap {
    OptionsMap map;
    this->gather_options(map);
    return map;
  };
  // Ensure that his own options are registered
  void IOption::ensure_initialized() {
    if (!m_init_ended) {
      m_init_phase = true;
      register_options();
      register_options_dependencies();
      m_init_phase = false;
      m_init_ended = true;
    }
  }
  void IOption::register_consumer(IOption &consumer) {
    if (m_init_phase) {
      m_consumers.push_back(&consumer);
    }
  };

} // namespace Baseliner