#include <baseliner/Options.hpp>
#include <iostream>
#include <stdexcept>
namespace Baseliner {
  namespace OptionBindings {

    void IOptionBinding::set_sweep_hint(const SweepHint &hint) {
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

  auto IOption::get_options() -> OptionsMap {
    OptionsMap omap;
    this->get_options(omap);
    return omap;
  };
  void IOption::get_options(OptionsMap &omap) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      omap[binding->get_interface_name()][binding->get_name()] =
          Option{binding->get_description(), binding->get_value()};
    }
  };
  auto IOption::get_sweep_hints() -> SweepHintMap {
    SweepHintMap hintmap;
    this->get_sweep_hints(hintmap);
    return hintmap;
  };

  void IOption::get_sweep_hints(SweepHintMap &hintmap) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      auto opt_hint = binding->get_sweep_hint();
      if (opt_hint.has_value()) {
        hintmap[binding->get_interface_name()][binding->get_name()] = opt_hint.value();
      }
    }
  }

  auto IOption::get_depedencies_options() -> OptionsMap {
    OptionsMap omap;
    this->get_depedencies_options(omap);
    return omap;
  };
  void IOption::get_depedencies_options(OptionsMap &omap) {
    std::unordered_set<IOption *> visited;
    return this->get_depedencies_options(omap, visited);
  }
  void IOption::get_depedencies_options(OptionsMap &omap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    get_options(omap);
    visited.insert(this);
    for (IOption *consumer : m_consumers) {
      consumer->get_depedencies_options(omap, visited);
    }
  }
  auto IOption::get_depedencies_sweep_hints() -> SweepHintMap {
    SweepHintMap hintmap;
    this->get_depedencies_sweep_hints(hintmap);
    return hintmap;
  };
  void IOption::get_depedencies_sweep_hints(SweepHintMap &hintmap) {
    std::unordered_set<IOption *> visited;
    return this->get_depedencies_sweep_hints(hintmap, visited);
  };

  void IOption::get_depedencies_sweep_hints(SweepHintMap &hintmap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    get_sweep_hints(hintmap);
    visited.insert(this);
    for (IOption *consumer : m_consumers) {
      consumer->get_depedencies_sweep_hints(hintmap, visited);
    }
  };

  void IOption::apply_options(const OptionsMap &omap) {
    for (auto &binding : m_options_bindings) {
      if (omap.find(binding->get_interface_name()) != omap.end()) {
        auto intermediary = omap.at(binding->get_interface_name());
        if (intermediary.find(binding->get_name()) != intermediary.end()) {
          const Option opt = intermediary.at(binding->get_name());
          binding->update_value(opt.m_value);
        }
      }
    }
  };
  void IOption::apply_depedencies_options(const OptionsMap &omap) {
    std::unordered_set<IOption *> visited;
    return this->apply_depedencies_options(omap, visited);
  };
  void IOption::apply_depedencies_options(const OptionsMap &omap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    this->apply_options(omap);
    for (IOption *consumer : m_consumers) {
      consumer->apply_depedencies_options(omap, visited);
    }
  }
  void IOption::update_sweep_hints(const SweepHintMap &hintmaps) {
    for (auto &binding : m_options_bindings) {
      if (hintmaps.find(binding->get_interface_name()) != hintmaps.end()) {
        auto intermediary = hintmaps.at(binding->get_interface_name());
        if (intermediary.find(binding->get_name()) != intermediary.end()) {
          const SweepHint hint = intermediary.at(binding->get_name());
          binding->set_sweep_hint(hint);
        }
      }
    }
  };

  void IOption::update_depedency_sweep_hints(const SweepHintMap &hintmaps) {
    std::unordered_set<IOption *> visited;
    return this->update_depedency_sweep_hints(hintmaps, visited);
  };
  void IOption::update_depedency_sweep_hints(const SweepHintMap &hintmaps, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    this->update_sweep_hints(hintmaps);
    for (IOption *consumer : m_consumers) {
      consumer->update_depedency_sweep_hints(hintmaps, visited);
    }
  };

  auto IOption::resolve_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector) -> std::vector<ResolvedAxis> {
    std::vector<ResolvedAxis> resolved;
    for (auto &axis : sweep_axis_vector) {
      for (auto &binding : m_options_bindings) {
        if (binding->get_name() == axis.m_option && binding->get_interface_name() == axis.m_interface) {
          binding->set_sweep_hint(axis.m_hint);
          resolved.emplace_back(axis.m_option, axis.m_interface, binding->generate_sweep_values());
        }
      }
    }
    return resolved;
  };

  auto IOption::resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector)
      -> std::vector<ResolvedAxis> {
    std::unordered_set<IOption *> visited;
    return this->resolve_depedency_sweep_axis(sweep_axis_vector, visited);
  };
  auto IOption::resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector,
                                             std::unordered_set<IOption *> &visited) -> std::vector<ResolvedAxis> {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    std::vector<ResolvedAxis> resolved_axis = this->resolve_depedency_sweep_axis(sweep_axis_vector);
    for (IOption *consumer : m_consumers) {
      auto temp_resolved = consumer->resolve_depedency_sweep_axis(sweep_axis_vector, visited);
      for (const auto &temp_res : temp_resolved) {
        for (const auto &res : resolved_axis) {
          if (temp_res.m_interface == res.m_interface && temp_res.m_option == res.m_option) {
            throw Errors::multiple_axis_responder(res);
          }
        }
      }
      resolved_axis.insert(resolved_axis.end(), temp_resolved.begin(), temp_resolved.end());
    }
    return resolved_axis;
  };
  // Ensure that his own options are registered and up to date
  void IOption::ensure_initialized() {
    m_init_phase = true;
    m_options_bindings.clear();
    m_consumers.clear();
    register_options();
    register_options_dependencies();
    m_init_phase = false;
  }
  void IOption::register_consumer(IOption *consumer) {
    if (m_init_phase) {
      if (consumer == this) {
        throw Errors::self_consumer(typeid(*this).name());
      }
      m_consumers.push_back(consumer);
    }
  };

} // namespace Baseliner