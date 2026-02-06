#ifndef OPTIONS_HPP
#define OPTIONS_HPP
#include <baseliner/Conversions.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Baseliner {
  // The OptionBinding Namespace is here to take care of the serialization and de-serialization of options
  // So that new options can propagate without having to write the code.
  namespace OptionBindings {
    struct OptionBindingBase {
      std::string m_interface_name;
      std::string m_name;
      std::string m_description;
      OptionBindingBase(std::string interface_name, std::string name, std::string description)
          : m_interface_name(interface_name),
            m_name(name),
            m_description(description) {};
      virtual ~OptionBindingBase() = default;
      virtual void update_value(const std::string &val) = 0;
      virtual std::string get_value() const = 0;
    };
    template <typename T>
    struct OptionBinding : public OptionBindingBase {
      T *m_val_ptr;
      OptionBinding(std::string interface_name, std::string name, std::string description, T &var)
          : OptionBindingBase(interface_name, name, description),
            m_val_ptr(&var) {};
      void update_value(const std::string &val) override {
        *m_val_ptr = Conversion::baseliner_from_string<T>(val);
      };
      std::string get_value() const override {
        return Conversion::baseliner_to_string(*m_val_ptr);
      };
    };
  } // namespace OptionBindings

  struct Option {
    std::string m_description;
    std::string m_value;
  };

  using InterfaceOptions = std::unordered_map<std::string, Option>;
  using OptionsMap = std::unordered_map<std::string, InterfaceOptions>;

  inline void mergeOptionsMap(OptionsMap &destination, const OptionsMap &source) {
    destination.insert(source.begin(), source.end());
  };

  class OptionConsumer {
  public:
    virtual void register_options() = 0;
    const OptionsMap describe_options();

    void apply_options(const OptionsMap &options_map);
    virtual ~OptionConsumer() = default;

  protected:
    virtual void on_update() {};
    template <typename T>
    void add_option(std::string interface, std::string name, std::string description, T &variable) {
      m_options_bindings.push_back(std::make_unique<OptionBindings::OptionBinding<T>>(
          std::move(interface), std::move(name), std::move(description), variable));
    }

  private:
    std::vector<std::unique_ptr<OptionBindings::OptionBindingBase>> m_options_bindings;
    bool m_is_registered = false;
    void ensure_registered();
  };

  class OptionBroadcaster {
  public:
    void gather_options(OptionsMap &optionsMap);
    OptionsMap gather_options();
    void propagate_options(const OptionsMap &optionsMap);
    virtual void register_dependencies() {};
    virtual ~OptionBroadcaster() = default;

  protected:
    void register_consumer(OptionConsumer &consumer);

  private:
    bool m_is_init = false;
    std::vector<OptionConsumer *> m_consumers;
    void ensure_initialized();
  };

} // namespace Baseliner

#endif // OPTIONS_HPP