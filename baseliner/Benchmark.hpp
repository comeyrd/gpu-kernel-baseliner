#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Executable.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Runner.hpp>
#include <memory>
#include <string>
#include <vector>
namespace Baseliner {

  class SingleAxeBenchmark : public IOptionBroadcaster, public IExecutable {
  public:
    SingleAxeBenchmark(std::shared_ptr<IRunner> runner, Axe &Axe)
        : m_runner(runner),
          m_axe(Axe) {};

    auto run_all() -> std::vector<Result> override {
      std::vector<Result> results_v{};
      const OptionsMap baseMap;
      m_runner->gather_options();
      OptionsMap tempMap;
      for (const std::string &axe_val : m_axe.m_values) {
        tempMap = baseMap;
        tempMap[m_axe.m_interface_name][m_axe.m_option_name].m_value = axe_val;
        m_runner->propagate_options(tempMap);
        const Result result = m_runner->run();
        results_v.push_back(result);
      }
      return results_v;
    };
    void register_dependencies() override {
      register_consumer(*m_runner);
    };

  private:
    std::shared_ptr<IRunner> m_runner;
    Axe m_axe;
  };

} // namespace Baseliner
#endif // BENCHMARK_HPP