#ifndef OPTIONS_HPP
#define OPTIONS_HPP
#include "baseliner/Error.hpp"
#include <baseliner/AxeSweeping.hpp>
#include <baseliner/Conversions.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Baseliner {
  // The OptionBinding Namespace is here to take care of the serialization and de-serialization of options
  // So that new options can propagate without having to write the code.
  namespace OptionBindings {
    template <typename T>
    inline auto sweep_hint_to_typed(const SweepHint &hint) -> TypedSweepHint<T> {
      TypedSweepHint<T> typed{};
      typed.m_policy = hint.m_policy;
      if (typed.m_policy == SweepPolicy::Enumerated) {
        typed.m_enumerated = Conversion::baseliner_from_string<T>(hint.m_enumerated);
        return typed;
      }
      if (hint.m_step.empty()) {
        typed.m_step = {};
      } else {
        typed.m_step = Conversion::baseliner_from_string<T>(hint.m_step);
      }
      typed.m_max = Conversion::baseliner_from_string<T>(hint.m_max);
      typed.m_min = Conversion::baseliner_from_string<T>(hint.m_min);
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

      void set_sweep_hint(SweepHint hint);
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
          if (hint.m_policy == SweepPolicy::Enumerated) {
            return hint.m_enumerated;
          }
          return Conversion::baseliner_to_string<T>(Sweep::generate_sweep_values(sweep_hint_to_typed<T>(hint)));
        }
        throw Errors::sweeping_error(get_interface_name(), get_name());
      };

    private:
      T *m_val_ptr;
    };
  } // namespace OptionBindings

  struct Option {
    std::string m_description;
    std::string m_value;
  };

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;
  using SweepHintMap = std::unordered_map<std::string, std::unordered_map<std::string, SweepHint>>;
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
    static inline auto is_subset(const OptionsMap &base_map, const OptionsMap &small_subset) -> bool {
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
  } // namespace Options

  // TODO Setup checks so there is not a cyclic depedency between OptionConsumer
  // TODO that will make gather_options or propagate_option infinite recursive calls
  // This has a small impact on object footprint, but it could have even less if we want everything to be an option,
  // even if they don't give option atm
  // We would store a unique_ptr to a struct with all the members (the vectors etc) and then if a class REALLY uses
  // options, the struct would be malloc
  // and the memory usefull ?
  class IOption {
  public:
    void gather_options(OptionsMap &opts);
    auto gather_options() -> OptionsMap;
    void propagate_options(const OptionsMap &optionsMap);

    void gather_sweep_hints(SweepHintMap &hintmaps);
    auto gather_sweep_hints() -> SweepHintMap;
    void update_sweep_hint(const SweepHintMap &hintmaps);
    auto generate_sweep_values_for(const std::string &interface, const std::string &option) -> std::vector<std::string>;

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
    virtual void register_options() = 0;
    virtual void register_options_dependencies() {};

    void register_consumer(IOption &consumer);
    virtual void on_update() {};
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
} // namespace Baseliner

#endif // OPTIONS_HPP