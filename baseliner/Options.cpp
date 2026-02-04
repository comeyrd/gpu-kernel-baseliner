#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  // OptionConsumer
  void OptionConsumer::apply_options(const OptionsMap &options_map) {
    ensure_registered();
    for (const auto &[interface_name, options] : options_map) {
      for (const auto &[option_name, opt] : options) {
        for (auto &binding : m_options_bindings) {
          if (binding->m_name == option_name) {
            try {
              binding->update_value(opt.m_value);
            } catch (std::invalid_argument const &e) {
              std::cout << "Error while applying option : " << binding->m_interface_name << "." << option_name << "="
                        << opt.m_value << " Using default value : " << binding->get_value() << std::endl;
            }
          }
        }
      }
    }
    on_update();
  }

  const OptionsMap OptionConsumer::describe_options() {
    ensure_registered();
    OptionsMap opts;
    for (const auto &binding : m_options_bindings) {
      opts[binding->m_interface_name][binding->m_name] = Option{binding->m_description, binding->get_value()};
    }
    return opts;
  }

  void OptionConsumer::ensure_registered() {
    if (!m_is_registered) {
      register_options();
      m_is_registered = true;
    }
  }

  // Option Broadcaster

  void OptionBroadcaster::gather_options(OptionsMap &optionsMap) {
    ensure_initialized();
    for (OptionConsumer *consumer : m_consumers) {
      mergeOptionsMap(optionsMap, consumer->describe_options());
    }
  };

  void OptionBroadcaster::propagate_options(const OptionsMap &optionsMap) {
    ensure_initialized();
    for (OptionConsumer *consumer : m_consumers) {
      consumer->apply_options(optionsMap);
    }
  };

  void OptionBroadcaster::register_consumer(OptionConsumer &consumer) {
    m_consumers.push_back(&consumer);
  };

  void OptionBroadcaster::ensure_initialized() {
    if (!m_is_init) {
      register_dependencies();
      m_is_init = true;
    }
  }
} // namespace Baseliner