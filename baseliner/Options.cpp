#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  // IOptionConsumer
  // Apply the options to its own parameters
  void IOptionConsumer::propagate_options(const OptionsMap &options_map) {
    ensure_initialized();
    for (const auto &[interface_name, options] : options_map) {
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
    for (IOptionConsumer *consumer : m_consumers) {
      consumer->propagate_options(options_map);
    }
  }

  // Describe its own options
  void IOptionConsumer::gather_options(OptionsMap &opts) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      opts[binding->get_interface_name()][binding->get_name()] =
          Option{binding->get_description(), binding->get_value()};
    }
    for (IOptionConsumer *consumer : m_consumers) {
      consumer->gather_options(opts);
    }
  }

  auto IOptionConsumer::gather_options() -> OptionsMap {
    OptionsMap map;
    this->gather_options(map);
    return map;
  };
  // Ensure that his own options are registered
  void IOptionConsumer::ensure_initialized() {
    if (!m_is_init) {
      register_options();
      register_dependencies();
      m_is_init = true;
    }
  }
  void IOptionConsumer::register_consumer(IOptionConsumer &consumer) {
    m_consumers.push_back(&consumer);
  };

} // namespace Baseliner