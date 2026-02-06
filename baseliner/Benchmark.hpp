#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Executable.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/research_questions/research_questions.hpp>

namespace Baseliner {

  class SingleAxeBenchmark : public OptionConsumer, public OptionBroadcaster, public Executable {
  public:
    SingleAxeBenchmark(RunnerBase &runner, Axe &Axe)
        : m_runner(runner),
          m_axe(Axe) {};

    std::vector<Result> run_all() {
      std::vector<Result> results_v{};
      OptionsMap baseMap;
      m_runner.gather_options();
      OptionsMap tempMap;
      for (AxeValue &axe_val : m_axe.m_values) {
        tempMap = baseMap;
        tempMap[m_axe.m_interface_name][m_axe.m_option_name].m_value = axe_val.m_value;
        m_runner.propagate_options(tempMap);
        Result result = m_runner.run();
        results_v.push_back(m_runner.run());
      }
      return results_v;
    };

  private:
    RunnerBase &m_runner;
    Axe m_axe;
  };

} // namespace Baseliner
#endif // BENCHMARK_HPP