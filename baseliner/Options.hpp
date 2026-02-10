#ifndef OPTIONS_HPP
#define OPTIONS_HPP
#include <baseliner/Conversions.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Baseliner {
  // The OptionBinding Namespace is here to take care of the serialization and de-serialization of options
  // So that new options can propagate without having to write the code.
  namespace OptionBindings {
    class IOptionBinding {
    public:
      IOptionBinding(std::string interface_name, std::string name, std::string description)
          : m_interface_name(std::move(interface_name)),
            m_name(std::move(name)),
            m_description(std::move(description)) {};
      virtual ~IOptionBinding() = default;
      virtual void update_value(const std::string &val) = 0;
      [[nodiscard]] virtual auto get_value() const -> std::string = 0;
      [[nodiscard]] auto get_name() const -> std::string {
        return m_name;
      };
      [[nodiscard]] auto get_interface_name() const -> std::string {
        return m_interface_name;
      };
      [[nodiscard]] auto get_description() const -> std::string {
        return m_description;
      };

    private:
      std::string m_interface_name;
      std::string m_name;
      std::string m_description;
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

    private:
      T *m_val_ptr;
    };
    template <typename T>
    class OptionBinding<std::vector<T>> : public IOptionBinding {
    public:
      OptionBinding(const std::string &interface_name, const std::string &name, const std::string &description,
                    std::vector<T> &var)
          : IOptionBinding(interface_name, name, description),
            m_val_ptr(&var) {};

      void update_value(const std::string &val) override {
        *m_val_ptr = Conversion::baseliner_vector_from_string<T>(val);
      };

      [[nodiscard]] auto get_value() const -> std::string override {
        return Conversion::baseliner_to_string(*m_val_ptr);
      };

    private:
      std::vector<T> *m_val_ptr;
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

  // TODO Setup checks so there is not a cyclic depedency between OptionConsumer
  // TODO that will make gather_options or propagate_option infinite recursive calls
  class IOptionConsumer {
  public:
    virtual void register_options() = 0;
    void gather_options(OptionsMap &opts);
    auto gather_options() -> OptionsMap;
    void propagate_options(const OptionsMap &optionsMap);
    virtual void register_dependencies() {};
    virtual ~IOptionConsumer() = default;
    IOptionConsumer() = default;

    IOptionConsumer(const IOptionConsumer &) = default;
    auto operator=(const IOptionConsumer &) -> IOptionConsumer & = default;

    // Explicitly allow moving (needed for vectors to work)
    IOptionConsumer(IOptionConsumer &&) = default;
    auto operator=(IOptionConsumer &&) -> IOptionConsumer & = default;

  protected:
    void register_consumer(IOptionConsumer &consumer);
    virtual void on_update() {};
    template <typename T>
    void add_option(const std::string &interface, const std::string &name, const std::string &description,
                    T &variable) {
      m_options_bindings.push_back(
          std::make_unique<OptionBindings::OptionBinding<T>>(interface, name, description, variable));
    }

  private:
    std::vector<std::unique_ptr<OptionBindings::IOptionBinding>> m_options_bindings;
    bool m_is_init = false;
    std::vector<IOptionConsumer *> m_consumers;
    void ensure_initialized();
  };
} // namespace Baseliner

#endif // OPTIONS_HPP