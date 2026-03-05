#ifndef BASELINER_SINGLE_AXE_SUITE_HPP
#define BASELINER_SINGLE_AXE_SUITE_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/State.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/managers/RegisteringMacros.hpp>

#include <iomanip>
namespace Baseliner {
  enum class TargetMap {
    None,
    Benchmark,
    Case
  };
  class SingleAxeSuite : public ISuite {
  public:
    SingleAxeSuite() = default;
    auto name() -> std::string override {
      return "SingleAxeSuite";
    }
    [[nodiscard]] auto run_all() -> RunResult override {
      const std::string &interface_name = m_axe.get_interface_name();
      const std::string &option_name = m_axe.get_option_name();

      if (interface_name.empty()) {
        throw std::runtime_error("Error: Suite launched without axes");
      }

      const OptionsMap basemap = get_benchmark()->gather_axe_options();

      TargetMap target = TargetMap::None;
      if (auto it = basemap.find(interface_name);
          it != basemap.end() && it->second.find(option_name) != it->second.end()) {
        target = TargetMap::Benchmark;
      } else {
        throw std::runtime_error("Axe error: No option " + option_name + " found in interface " + interface_name +
                                 " for either bench or case setups.");
      }
      std::vector<BenchmarkResult> results_v{};
      results_v.reserve(m_axe.get_values().size());
      bool first = true;
      for (const std::string &axe_val : m_axe.get_values()) {
        OptionsMap patch_option;
        patch_option[interface_name][option_name].m_value = axe_val;
        get_benchmark()->propagate_axe_options(patch_option);

        BenchmarkResult result = get_benchmark()->run();
        result.m_options = patch_option;
        results_v.push_back(result);
        print_benchmark_result(std::cout, result, first);
        if (first) {
          first = false;
        }
        if (ExecutionController::exit_requested()) {
          break;
        }
      }
      return build_run_result(results_v, get_benchmark()->get_device_info());
    }

    void register_options_dependencies() override {
      this->register_consumer(m_axe);
    };

  private:
    SingleAxe m_axe;
    bool m_first_result = true;
    std::vector<int> m_col_widths;
    std::vector<std::string> m_metric_names;
  };
} // namespace Baseliner
#endif // BASELINER_SINGLE_AXE_SUITE_HPP