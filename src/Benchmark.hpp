#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include "Durations.hpp"
#include "Options.hpp"
#include "Runner.hpp"

#include <utility>
#include <vector>
namespace Baseliner {

  struct Axe {
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<std::string> m_values;
  };

  std::vector<OptionsMap> generate_permutations(OptionsMap base, const std::vector<Axe> &axes, int current = 0) {
    const Axe &axe = axes[current];
    current++;
    std::vector<OptionsMap> omaps;
    for (std::string value : axe.m_values) {
      OptionsMap inner_om = base;
      inner_om[axe.m_interface_name][axe.m_option_name].m_value = value;
      if (current < axes.size()) {
        auto getted_omaps = generate_permutations(inner_om, axes, current);
        omaps.insert(omaps.end(), getted_omaps.begin(), getted_omaps.end());
      } else {
        omaps.push_back(inner_om);
      }
    }
    return omaps;
  };

  class BareBenchmark {
  public:
    BareBenchmark(RunnerBase &runner)
        : m_runner(runner) {};

    std::vector<std::pair<OptionsMap, std::vector<float_milliseconds>>> run(std::vector<Axe> &axes) {
      OptionsMap omap;
      m_runner.gather_options(omap);
      std::vector<OptionsMap> omap_v = generate_permutations(omap, axes);
      std::vector<std::pair<OptionsMap, std::vector<float_milliseconds>>> result_vector;
      for (OptionsMap &omap : omap_v) {
        m_runner.apply_options(omap);
        std::vector<float_milliseconds> measures = m_runner.run();
        result_vector.push_back(std::make_pair(omap, measures));
      }
      return result_vector;
    };

  private:
    RunnerBase &m_runner;
  };

} // namespace Baseliner
#endif // BENCHMARK_HPP