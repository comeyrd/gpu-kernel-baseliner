#ifndef BASELINER_PRESET_INJECTION_HPP
#define BASELINER_PRESET_INJECTION_HPP
#include "baseliner/Benchmark.hpp"
#include <baseliner/Options.hpp>
#include <baseliner/Serializer.hpp>
#include <functional>
#include <memory>
namespace Baseliner {
  inline auto inject_option(const std::function<std::shared_ptr<IBenchmark>()> &funct,
                            const OptionsMap &benchmark_preset, const OptionsMap &case_preset,
                            const OptionsMap &stat_options) -> std::function<std::shared_ptr<IBenchmark>()> {

    auto output_function = [funct, benchmark_preset, case_preset, stat_options]() -> std::shared_ptr<IBenchmark> {
      auto ptr = funct();
      auto benchmark_options = ptr->get_options();
      if (Options::is_subset(benchmark_options, benchmark_preset)) {
        ptr->apply_options(benchmark_preset);
      } else {
        throw Errors::preset_not_subset_of(benchmark_preset, benchmark_options);
      }
      auto case_options = ptr->get_case_options();
      if (Options::is_subset(case_options, case_preset)) {
        ptr->apply_options(case_preset);
      } else {
        throw Errors::preset_not_subset_of(case_preset, case_options);
      }
      ptr->set_stat_options(stat_options);
      return ptr;
    };
    return output_function;
  }

  inline auto inject_option(const std::function<std::unique_ptr<StoppingCriterion>()> &funct, const OptionsMap &preset)
      -> std::function<std::unique_ptr<StoppingCriterion>()> {
    static_assert(std::is_base_of_v<IOption, StoppingCriterion>,
                  "The type you want to inject presets on must inherit from IOption");
    auto output_function = [funct, preset]() -> std::unique_ptr<StoppingCriterion> {
      auto ptr = funct();
      auto options = ptr->get_options();
      if (Options::is_subset(options, preset)) {
        ptr->apply_options(preset);
      } else {
        throw Errors::preset_not_subset_of(preset, options);
      }
      return ptr;
    };
    return output_function;
  }
} // namespace Baseliner
#endif // BASELINER_PRESET_INJECTION_HPP