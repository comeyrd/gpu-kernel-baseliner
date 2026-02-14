#ifndef BASELINER_SUITE_HPP
#define BASELINER_SUITE_HPP
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Task.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace Baseliner {

  class SingleAxeSuite : public IOption, public ITask {
  public:
    SingleAxeSuite(std::shared_ptr<IBenchmark> benchmark, Axe Axe)
        : m_benchmark(std::move(benchmark)),
          m_axe(std::move(Axe)) {};

    auto run_all() -> std::vector<Result> override {
      std::vector<Result> results_v{};
      const OptionsMap baseMap;
      m_benchmark->gather_options();
      OptionsMap tempMap;
      for (const std::string &axe_val : m_axe.get_values()) {
        tempMap = baseMap;
        tempMap[m_axe.get_interface_name()][m_axe.get_option_name()].m_value = axe_val;
        m_benchmark->propagate_options(tempMap);
        const Result result = m_benchmark->run();
        results_v.push_back(result);
      }
      return results_v;
    };

  protected:
    void register_options_dependencies() override {
      register_consumer(*m_benchmark);
      register_consumer(m_axe);
    };
    void register_options() override {
    }

  private:
    std::shared_ptr<IBenchmark> m_benchmark;
    Axe m_axe;
  };

} // namespace Baseliner
#endif // BASELINER_SUITE_HPP