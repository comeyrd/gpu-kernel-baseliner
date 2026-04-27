#ifndef BASELINER_CORE_OPTIONS_HPP
#define BASELINER_CORE_OPTIONS_HPP
#include <baseliner/specs/AxeSweeping.hpp>
#include <baseliner/specs/Conversions.hpp>

#include <baseliner/specs/Error.hpp>
#include <baseliner/specs/OptionTypes.hpp>

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Baseliner {
  // The OptionBinding Namespace is here to take care of the serialization and de-serialization of options
  // So that new options can propagate without having to write the code.
  namespace OptionBindings {
    template <typename T>
    inline auto sweep_hint_to_typed(const SweepHint &hint) -> TypedSweepHint<T> {
      TypedSweepHint<T> typed{};
      typed.policy = hint.policy;
      if (typed.policy == SweepPolicy::Enumerated) {
        typed.enumerated = Conversion::baseliner_from_string<T>(hint.enumerated);
        return typed;
      }
      if (hint.step.empty()) {
        typed.step = T{};
      } else {
        typed.step = Conversion::baseliner_from_string<T>(hint.step);
      }
      typed.max = Conversion::baseliner_from_string<T>(hint.max);
      typed.min = Conversion::baseliner_from_string<T>(hint.min);
      return typed;
    };
    class IOptionBinding {
    public:
      IOptionBinding(std::string interface_name, std::string name, std::string description)
          : m_interface_name(std::move(interface_name)),
            m_name(std::move(name)),
            m_description(std::move(description)) {};
      virtual ~IOptionBinding() = default;
      virtual void update_value(const std::string &val) = 0;
      [[nodiscard]] virtual auto get_value() const -> std::string = 0;
      [[nodiscard]] auto get_name() const -> std::string;
      [[nodiscard]] auto get_interface_name() const -> std::string;
      [[nodiscard]] auto get_description() const -> std::string;

      void set_sweep_hint(const SweepHint &hint);
      [[nodiscard]] auto get_sweep_hint() const -> const std::optional<SweepHint> &;
      [[nodiscard]] auto is_sweepable() const -> bool;
      [[nodiscard]] virtual auto generate_sweep_values() const -> std::vector<std::string> = 0;

    private:
      std::string m_interface_name;
      std::string m_name;
      std::string m_description;
      std::optional<SweepHint> m_sweep_hint;
    };

    template <typename T>
    class OptionBinding : public IOptionBinding {
    public:
      OptionBinding(const std::string &interface_name, const std::string &name, const std::string &description, T &var)
          : IOptionBinding(interface_name, name, description),
            m_val_ptr(&var) {};
      void update_value(const std::string &val) override {
        *m_val_ptr = Conversion::baseliner_from_string<T>(val);
      };
      [[nodiscard]] auto get_value() const -> std::string override {
        return Conversion::baseliner_to_string(*m_val_ptr);
      };
      auto sweep(SweepPolicy policy, const std::string &min, const std::string &max, const std::string &step = "")
          -> OptionBinding<T> & {
        set_sweep_hint(SweepHint{policy, min, max, step, {}});
        return *this;
      }
      auto sweep(const std::vector<std::string> &values) -> OptionBinding<T> & {
        set_sweep_hint(SweepHint{SweepPolicy::Enumerated, "", "", "", values});
        return *this;
      }

      [[nodiscard]] auto generate_sweep_values() const -> std::vector<std::string> override {
        if (this->is_sweepable()) {
          SweepHint hint = this->get_sweep_hint().value();
          if (hint.policy == SweepPolicy::Enumerated) {
            return hint.enumerated;
          }
          return Conversion::baseliner_to_string<T>(Sweep::generate_sweep_values(sweep_hint_to_typed<T>(hint)));
        }
        throw Errors::sweeping_error(get_interface_name(), get_name());
      };

    private:
      T *m_val_ptr;
    };
  } // namespace OptionBindings

  namespace Options {
    static inline auto have_same_schema(const OptionsMap &omap1, const OptionsMap &omap2) -> bool {
      if (omap1.size() != omap2.size()) {
        return false;
      }
      for (const auto &[interface_name, interface_opt] : omap1) {
        auto omap2_interface_it = omap2.find(interface_name);
        if (omap2_interface_it == omap2.end()) {
          return false;
        }
        const auto &interface_opt_omap2 = omap2_interface_it->second;
        if (interface_opt.size() != interface_opt_omap2.size()) {
          return false;
        }
        for (const auto &[option_name, _] : interface_opt) {
          if (interface_opt_omap2.find(option_name) == interface_opt_omap2.end()) {
            return false;
          }
        }
      }
      return true;
    }
    // True if small_subset is a subset of base_map
    inline auto is_subset(const OptionsMap &base_map, const OptionsMap &small_subset) -> bool {
      for (const auto &[interface_name, interface_opt] : small_subset) {
        auto omap2_interface_it = base_map.find(interface_name);
        if (omap2_interface_it == base_map.end()) {
          return false;
        }
        const auto &interface_opt_omap2 = omap2_interface_it->second;
        for (const auto &[option_name, _] : interface_opt) {
          if (interface_opt_omap2.find(option_name) == interface_opt_omap2.end()) {
            return false;
          }
        }
      }
      return true;
    }
    // Returns a new OptionsMap with all the values of "overrides" and the values from "base" which overrides misses.
    inline auto merge(const OptionsMap &base, const OptionsMap &overrides) -> OptionsMap {
      OptionsMap result = base;
      for (const auto &[key, value] : overrides) {
        result[key] = value;
      }
      return result;
    }
  } // namespace Options

  // TODO Setup checks so there is not a cyclic depedency between OptionConsumer
  // TODO that will make get_options or propagate_option infinite recursive calls
  // This has a small impact on object footprint, but it could have even less if we want everything to be an option,
  // even if they don't give option atm
  // We would store a unique_ptr to a struct with all the members (the vectors etc) and then if a class REALLY uses
  // options, the struct would be malloc
  // and the memory usefull ?
  class IOption {
  public:
    auto get_options() -> OptionsMap;
    void get_options(OptionsMap &omap);

    auto get_sweep_hints() -> SweepHintMap;
    void get_sweep_hints(SweepHintMap &hintmap);

    auto get_depedencies_options() -> OptionsMap;
    void get_depedencies_options(OptionsMap &omap);

    auto get_depedencies_sweep_hints() -> SweepHintMap;
    void get_depedencies_sweep_hints(SweepHintMap &hintmap);

    void apply_options(const OptionsMap &omap);
    void apply_depedencies_options(const OptionsMap &omap);

    void update_sweep_hints(const SweepHintMap &hintmaps);
    void update_depedency_sweep_hints(const SweepHintMap &hintmaps);

    [[nodiscard]] auto resolve_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector) -> std::vector<ResolvedAxis>;
    [[nodiscard]] auto resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector)
        -> std::vector<ResolvedAxis>;

    virtual ~IOption() = default;
    IOption() = default;

    IOption(const IOption &) = delete;
    auto operator=(const IOption &) -> IOption & = delete;

    // Moving
    IOption(IOption &&other) noexcept {
      other.m_consumers.clear();
      other.m_options_bindings.clear();
      other.m_init_ended = false;
      other.m_init_phase = false;
    }

    // Moving
    auto operator=(IOption &&other) noexcept -> IOption & {
      if (this != &other) {
        this->m_consumers.clear();
        this->m_options_bindings.clear();
        this->m_init_phase = false;
        this->m_init_ended = false;
        this->m_options_bindings.clear();
        other.m_consumers.clear();
        other.m_options_bindings.clear();
        other.m_init_ended = false;
        other.m_init_phase = false;
      }
      return *this;
    }

  protected:
    void get_depedencies_options(OptionsMap &omap, std::unordered_set<IOption *> &visited);
    void get_depedencies_sweep_hints(SweepHintMap &hintmap, std::unordered_set<IOption *> &visited);
    void apply_depedencies_options(const OptionsMap &omap, std::unordered_set<IOption *> &visited);
    void update_depedency_sweep_hints(const SweepHintMap &hintmaps, std::unordered_set<IOption *> &visited);
    [[nodiscard]] auto resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector,
                                                    std::unordered_set<IOption *> &visited)
        -> std::vector<ResolvedAxis>;

    virtual void register_options() = 0;
    virtual void register_options_dependencies() {};

    void register_consumer(IOption *consumer);
    virtual void on_update() {}; // Called when the options are updated
    template <typename T>
    auto add_option(const std::string &interface, const std::string &name, const std::string &description, T &variable)
        -> OptionBindings::OptionBinding<T> & {
      if (m_init_phase) {
        auto binding = std::make_unique<OptionBindings::OptionBinding<T>>(interface, name, description, variable);
        auto &ref = *binding;
        m_options_bindings.push_back(std::move(binding));
        return ref;
      }
      throw Errors::adding_option_outside_register_option();
    }

  private:
    bool m_init_ended = false;
    bool m_init_phase = false;
    // Never access these two vectors to add bindings or consumer else than with ensure_initialized();
    std::vector<std::unique_ptr<OptionBindings::IOptionBinding>> m_options_bindings;
    std::vector<IOption *> m_consumers;
    void ensure_initialized();
  };

  class LazyOption : public IOption {
  public:
    void register_options() override {};
  };

  //////// IMPL ///////////////////////
  // Fix for "Spec" version of baseliner

  namespace OptionBindings {

    inline void IOptionBinding::set_sweep_hint(const SweepHint &hint) {
      m_sweep_hint = hint;
    };
    [[nodiscard]] inline auto IOptionBinding::get_sweep_hint() const -> const std::optional<SweepHint> & {
      return m_sweep_hint;
    };
    [[nodiscard]] inline auto IOptionBinding::is_sweepable() const -> bool {
      return m_sweep_hint.has_value();
    };
    [[nodiscard]] inline auto IOptionBinding::get_name() const -> std::string {
      return m_name;
    };
    [[nodiscard]] inline auto IOptionBinding::get_interface_name() const -> std::string {
      return m_interface_name;
    };
    [[nodiscard]] inline auto IOptionBinding::get_description() const -> std::string {
      return m_description;
    };
  } // namespace OptionBindings

  // IOption

  inline auto IOption::get_options() -> OptionsMap {
    OptionsMap omap;
    this->get_options(omap);
    return omap;
  };
  inline void IOption::get_options(OptionsMap &omap) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      omap[binding->get_interface_name()][binding->get_name()] =
          Option{binding->get_description(), binding->get_value()};
    }
  };
  inline auto IOption::get_sweep_hints() -> SweepHintMap {
    SweepHintMap hintmap;
    this->get_sweep_hints(hintmap);
    return hintmap;
  };

  inline void IOption::get_sweep_hints(SweepHintMap &hintmap) {
    ensure_initialized();
    for (const auto &binding : m_options_bindings) {
      auto opt_hint = binding->get_sweep_hint();
      if (opt_hint.has_value()) {
        hintmap[binding->get_interface_name()][binding->get_name()] = opt_hint.value();
      }
    }
  }

  inline auto IOption::get_depedencies_options() -> OptionsMap {
    OptionsMap omap;
    this->get_depedencies_options(omap);
    return omap;
  };
  inline void IOption::get_depedencies_options(OptionsMap &omap) {
    std::unordered_set<IOption *> visited;
    this->get_depedencies_options(omap, visited);
  }
  inline void IOption::get_depedencies_options(OptionsMap &omap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    get_options(omap);
    visited.insert(this);
    for (IOption *consumer : m_consumers) {
      consumer->get_depedencies_options(omap, visited);
    }
  }
  inline auto IOption::get_depedencies_sweep_hints() -> SweepHintMap {
    SweepHintMap hintmap;
    this->get_depedencies_sweep_hints(hintmap);
    return hintmap;
  };
  inline void IOption::get_depedencies_sweep_hints(SweepHintMap &hintmap) {
    std::unordered_set<IOption *> visited;
    this->get_depedencies_sweep_hints(hintmap, visited);
  };

  inline void IOption::get_depedencies_sweep_hints(SweepHintMap &hintmap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    get_sweep_hints(hintmap);
    visited.insert(this);
    for (IOption *consumer : m_consumers) {
      consumer->get_depedencies_sweep_hints(hintmap, visited);
    }
  };

  inline void IOption::apply_options(const OptionsMap &omap) {
    this->ensure_initialized();
    for (auto &binding : m_options_bindings) {
      if (omap.find(binding->get_interface_name()) != omap.end()) {
        auto intermediary = omap.at(binding->get_interface_name());
        if (intermediary.find(binding->get_name()) != intermediary.end()) {
          const Option opt = intermediary.at(binding->get_name());
          binding->update_value(opt.value);
        }
      }
    }
    on_update();
  };
  inline void IOption::apply_depedencies_options(const OptionsMap &omap) {
    std::unordered_set<IOption *> visited;
    this->apply_depedencies_options(omap, visited);
  };
  inline void IOption::apply_depedencies_options(const OptionsMap &omap, std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    this->apply_options(omap);
    for (IOption *consumer : m_consumers) {
      consumer->apply_depedencies_options(omap, visited);
    }
  }
  inline void IOption::update_sweep_hints(const SweepHintMap &hintmaps) {
    this->ensure_initialized();
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

  inline void IOption::update_depedency_sweep_hints(const SweepHintMap &hintmaps) {
    std::unordered_set<IOption *> visited;
    this->update_depedency_sweep_hints(hintmaps, visited);
  };
  inline void IOption::update_depedency_sweep_hints(const SweepHintMap &hintmaps,
                                                    std::unordered_set<IOption *> &visited) {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    this->update_sweep_hints(hintmaps);
    for (IOption *consumer : m_consumers) {
      consumer->update_depedency_sweep_hints(hintmaps, visited);
    }
  };

  inline auto IOption::resolve_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector)
      -> std::vector<ResolvedAxis> {
    std::vector<ResolvedAxis> resolved;
    for (const auto &axis : sweep_axis_vector) {
      for (auto &binding : m_options_bindings) {
        if (binding->get_name() == axis.option && binding->get_interface_name() == axis.interface) {
          if (axis.hint.has_value()) {
            binding->set_sweep_hint(axis.hint.value());
          }
          resolved.push_back(ResolvedAxis{axis.interface, axis.option, binding->generate_sweep_values()});
        }
      }
    }
    return resolved;
  };

  inline auto IOption::resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector)
      -> std::vector<ResolvedAxis> {
    std::unordered_set<IOption *> visited;
    return this->resolve_depedency_sweep_axis(sweep_axis_vector, visited);
  };
  inline auto IOption::resolve_depedency_sweep_axis(const std::vector<SweepAxis> &sweep_axis_vector,
                                                    std::unordered_set<IOption *> &visited)
      -> std::vector<ResolvedAxis> {
    if (visited.find(this) != visited.end()) {
      throw Errors::recursive_consumer_options(typeid(*this).name());
    }
    visited.insert(this);
    std::vector<ResolvedAxis> resolved_axis = this->resolve_sweep_axis(sweep_axis_vector);
    for (IOption *consumer : m_consumers) {
      auto temp_resolved = consumer->resolve_depedency_sweep_axis(sweep_axis_vector, visited);
      for (const auto &temp_res : temp_resolved) {
        for (const auto &res : resolved_axis) {
          if (temp_res.interface == res.interface && temp_res.option == res.option) {
            throw Errors::multiple_axis_responder(res);
          }
        }
      }
      resolved_axis.insert(resolved_axis.end(), temp_resolved.begin(), temp_resolved.end());
    }
    return resolved_axis;
  };
  // Ensure that his own options are registered and up to date
  inline void IOption::ensure_initialized() {
    m_init_phase = true;
    m_options_bindings.clear();
    m_consumers.clear();
    register_options();
    register_options_dependencies();
    m_init_phase = false;
  }
  inline void IOption::register_consumer(IOption *consumer) {
    if (m_init_phase) {
      if (consumer == this) {
        throw Errors::self_consumer(typeid(*this).name());
      }
      m_consumers.push_back(consumer);
    }
  };

} // namespace Baseliner

#endif // OPTIONS_HPP