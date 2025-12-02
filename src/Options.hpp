#ifndef OPTIONS_HPP
#define OPTIONS_HPP
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Baseliner {
  inline std::string bool_to_string(bool value) {
    return std::to_string(static_cast<int>(value));
  };
  inline bool string_to_bool(std::string value) {
    int i_value = std::stoi(value);
    return (i_value != 0);
  };

  struct Option {
    std::string m_description;
    std::string m_value;
  };
  struct OptionBindingBase {
    std::string m_name;
    std::string m_description;
    OptionBindingBase(std::string name, std::string description)
        : m_name(name),
          m_description(description) {};
    virtual ~OptionBindingBase() = default;
    virtual void update_value(const std::string &val) = 0;
    virtual std::string get_value() const = 0;
  };
  template <typename T>
  struct OptionBinding : public OptionBindingBase {
    T *m_val_ptr;
    OptionBinding(std::string name, std::string description, T &var)
        : OptionBindingBase(name, description),
          m_val_ptr(&var) {};
    void update_value(const std::string &val) override;
    std::string get_value() const override {
      return std::to_string(*m_val_ptr);
    };
  };
  template <>
  inline void OptionBinding<int>::update_value(const std::string &val) {
    *m_val_ptr = std::stoi(val);
  };
  template <>
  inline void OptionBinding<float>::update_value(const std::string &val) {
    *m_val_ptr = std::stof(val);
  };
  template <>
  inline void OptionBinding<bool>::update_value(const std::string &val) {
    *m_val_ptr = string_to_bool(val);
  };
  template <>
  inline std::string OptionBinding<bool>::get_value() const {
    return bool_to_string(*m_val_ptr);
  };
  template <>
  inline void OptionBinding<std::string>::update_value(const std::string &val) {
    *m_val_ptr = val;
  };
  template <>
  inline std::string OptionBinding<std::string>::get_value() const {
    return *m_val_ptr;
  };

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;

  class OptionConsumer {
  public:
    virtual const std::string get_name() = 0;
    virtual void register_options() = 0;
    const InterfaceOptions describe_options() {
      ensure_registered();
      InterfaceOptions opts;
      for (const auto &binding : m_options_bindings) {
        opts[binding->m_name] = Option{binding->m_description, binding->get_value()};
      }
      return opts;
    }

    void apply_options(const InterfaceOptions &options) {
      ensure_registered();
      for (const auto &[name, opt] : options) {
        for (auto &binding : m_options_bindings) {
          if (binding->m_name == name) {
            try {
              binding->update_value(opt.m_value);
            } catch (std::invalid_argument const &e) {
              std::cout << "Error while applying option : " << get_name() << "." << name << "=" << opt.m_value
                        << " Using default value : " << binding->get_value() << std::endl;
            }
          }
        }
      }
      on_update();
    }

  protected:
    virtual void on_update() {};
    template <typename T>
    void add_option(std::string name, std::string description, T &variable) {
      m_options_bindings.push_back(
          std::make_unique<OptionBinding<T>>(std::move(name), std::move(description), variable));
    }

  private:
    std::vector<std::unique_ptr<OptionBindingBase>> m_options_bindings;
    bool m_is_registered = false;
    void ensure_registered() {
      if (!m_is_registered) {
        register_options();
        m_is_registered = true;
      }
    }
  };
  class OptionBroadcaster {
  public:
    void gather_options(OptionsMap &optionsMap) {
      ensure_initialized();
      for (OptionConsumer *consumer : m_consumers) {
        optionsMap[consumer->get_name()] = consumer->describe_options();
      }
    };
    void propagate_options(const OptionsMap &optionsMap) {
      ensure_initialized();
      for (OptionConsumer *consumer : m_consumers) {
        if (auto option = optionsMap.find(consumer->get_name()); option != optionsMap.end()) {
          consumer->apply_options(option->second);
        }
      }
    };
    virtual void register_dependencies() {};

  protected:
    void register_consumer(OptionConsumer &consumer) {
      m_consumers.push_back(&consumer);
    }

  private:
    bool m_is_init = false;
    std::vector<OptionConsumer *> m_consumers;
    void ensure_initialized() {
      if (!m_is_init) {
        register_dependencies();
        m_is_init = true;
      }
    }
  };

} // namespace Baseliner
#endif // OPTIONS_HPP