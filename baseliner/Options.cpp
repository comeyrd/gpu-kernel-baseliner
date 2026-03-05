#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  namespace OptionBindings {

    void IOptionBinding::set_sweep_hint(SweepHint hint) {
      m_sweep_hint = hint;
    };
    [[nodiscard]] auto IOptionBinding::get_sweep_hint() const -> const std::optional<SweepHint> & {
      return m_sweep_hint;
    };
    [[nodiscard]] auto IOptionBinding::is_sweepable() const -> bool {
      return m_sweep_hint.has_value();
    };
    [[nodiscard]] auto IOptionBinding::get_name() const -> std::string {
      return m_name;
    };
    [[nodiscard]] auto IOptionBinding::get_interface_name() const -> std::string {
      return m_interface_name;
    };
    [[nodiscard]] auto IOptionBinding::get_description() const -> std::string {
      return m_description;
    };
  } // namespace OptionBindings

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

  void IOption::gather_sweep_hints(SweepHintMap &hintmaps) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      auto sweephint = binding->get_sweep_hint();
      if (sweephint.has_value()) {
        hintmaps[binding->get_interface_name()][binding->get_name()] = sweephint.value();
      }
    }
    for (IOption *consumer : m_consumers) {
      consumer->gather_sweep_hints(hintmaps);
    }
  };
  auto IOption::gather_sweep_hints() -> SweepHintMap {
    SweepHintMap map;
    this->gather_sweep_hints(map);
    return map;
  };
  void IOption::update_sweep_hint(const SweepHintMap &hintmaps) {
    ensure_initialized();
    for (const auto &[interface_name, options] : hintmaps) {
      for (const auto &[option_name, sweepHint] : options) {
        for (auto &binding : m_options_bindings) {
          if (binding->get_name() == option_name && binding->get_interface_name() == interface_name) {
            binding->set_sweep_hint(sweepHint);
          }
        }
      }
    }
    on_update();
    for (IOption *consumer : m_consumers) {
      consumer->update_sweep_hint(hintmaps);
    }
  };
  auto IOption::generate_sweep_values_for(const std::string &interface, const std::string &option)
      -> std::vector<std::string> {
    std::vector<std::string> result = {};
    ensure_initialized();
    for (auto &binding : m_options_bindings) {
      if (binding->get_name() == option && binding->get_interface_name() == interface) {
        auto temp = binding->generate_sweep_values();
        if (!temp.empty()) {
          if (!result.empty()) {
            std::cerr << "Option Error : Two bindings have responded to generate sweep values" << interface << "."
                      << option << ", keeping the last one \n";
          }
          result = temp;
        }
      }
    }
    for (IOption *consumer : m_consumers) {
      auto temp = consumer->generate_sweep_values_for(interface, option);
      if (!temp.empty()) {
        if (!result.empty()) {
          std::cerr << "Option Error : Two bindings have responded to generate sweep values" << interface << "."
                    << option << ", keeping the last one \n";
        }
        result = temp;
      }
    }
    return result;
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