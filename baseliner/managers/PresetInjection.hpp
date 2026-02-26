#ifndef BASELINER_PRESET_INJECTION_HPP
#define BASELINER_PRESET_INJECTION_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Serializer.hpp>
#include <functional>
#include <memory>
namespace Baseliner {
  template <typename InnerTypeT>
  auto inject_shared_preset(const std::function<std::shared_ptr<InnerTypeT>()> &funct, const OptionsMap &preset)
      -> std::function<std::shared_ptr<InnerTypeT>()> {
    static_assert(std::is_base_of_v<IOption, InnerTypeT>,
                  "The type you want to inject presets on must inherit from IOption");
    auto output_function = [funct, preset]() -> std::shared_ptr<InnerTypeT> {
      auto ptr = funct();
      auto options = ptr->gather_options();
      if (Options::have_same_schema(options, preset)) {
        ptr->propagate_options(preset);
      } else {
        std::stringstream string_stream{};
        string_stream << "Error, presets should exactly match the object Option Schema \n";
        string_stream << "The given preset : \n";
        serialize(string_stream, preset);
        string_stream << "\n" << "The object preset \n";
        serialize(string_stream, options);
        std::runtime_error(string_stream.str());
      }
      return ptr;
    };
    return output_function;
  }
  template <typename InnerTypeT>
  auto inject_unique_preset(const std::function<std::unique_ptr<InnerTypeT>()> &funct, const OptionsMap &preset)
      -> std::function<std::unique_ptr<InnerTypeT>()> {
    static_assert(std::is_base_of_v<IOption, InnerTypeT>,
                  "The type you want to inject presets on must inherit from IOption");
    auto output_function = [funct, preset]() -> std::unique_ptr<InnerTypeT> {
      auto ptr = funct();
      auto options = ptr->gather_options();
      if (Options::have_same_schema(options, preset)) {
        ptr->propagate_options(preset);
      } else {
        std::stringstream string_stream{};
        string_stream << "Error, presets should exactly match the object Option Schema \n";
        string_stream << "The given preset : \n";
        serialize(string_stream, preset);
        string_stream << "\n" << "The object preset \n";
        serialize(string_stream, options);
        std::runtime_error(string_stream.str());
      }
      return ptr;
    };
    return output_function;
  }
} // namespace Baseliner
#endif // BASELINER_PRESET_INJECTION_HPP