#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  // IOptionConsumer
  // Apply the options to its own parameters
  void IOptionConsumer::apply_options(const OptionsMap &options_map) {
    ensure_registered();
    for (const auto &[interface_name, options] : options_map) {
      for (const auto &[option_name, opt] : options) {
        for (auto &binding : m_options_bindings) {
          if (binding->get_name() == option_name) {
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
  }
  // Describe its own options
  auto IOptionConsumer::describe_options() -> OptionsMap {
    ensure_registered();
    OptionsMap opts;
    for (const auto &binding : m_options_bindings) {
      opts[binding->get_interface_name()][binding->get_name()] =
          Option{binding->get_description(), binding->get_value()};
    }
    return opts;
  }
  // Ensure that his own options are registered
  void IOptionConsumer::ensure_registered() {
    if (!m_is_registered) {
      register_options();
      m_is_registered = true;
    }
  }

  // Option Broadcaster
  //
  void IOptionBroadcaster::gather_options(OptionsMap &optionsMap) {
    ensure_initialized();
    for (IOptionConsumer *consumer : m_consumers) {
      mergeOptionsMap(optionsMap, consumer->describe_options());
    }
  };
  auto IOptionBroadcaster::gather_options() -> OptionsMap {
    OptionsMap map;
    this->gather_options(map);
    return map;
  };

  void IOptionBroadcaster::propagate_options(const OptionsMap &optionsMap) {
    ensure_initialized();
    for (IOptionConsumer *consumer : m_consumers) {
      consumer->apply_options(optionsMap);
    }
  };

  void IOptionBroadcaster::register_consumer(IOptionConsumer &consumer) {
    m_consumers.push_back(&consumer);
  };

  void IOptionBroadcaster::ensure_initialized() {
    if (!m_is_init) {
      register_dependencies();
      m_is_init = true;
    }
  }
} // namespace Baseliner