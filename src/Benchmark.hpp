#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include "Durations.hpp"
#include "Options.hpp"
#include "Runner.hpp"
#include "research_questions/research_questions.hpp"
#include <utility>
#include <vector>
namespace Baseliner {

  struct AxeValue {
    std::string m_value;
    std::vector<float_milliseconds> m_results;
    AxeValue(std::string name)
        : m_value(name) {};
  };

  struct Axe {
    std::string m_interface_name;
    std::string m_option_name;
    std::vector<AxeValue> m_values;
  };

  class RqBenchmark {
  public:
    RqBenchmark(RunnerBase &runner, std::vector<ResearchQuestions::Question> &questions)
        : m_runner(runner),
          m_questions(questions) {};

    void run() {
      OptionsMap baseMap;
      m_runner.gather_options(baseMap);
      OptionsMap tempMap;
      for (Question &q : m_questions) {
        for (AxeValue &axe_val : q.m_axe.m_values) {
          tempMap = baseMap;
          tempMap[q.m_axe.m_interface_name][q.m_axe.m_option_name].m_value = axe_val.m_value;
          m_runner.propagate_options(tempMap);
          axe_val.m_results = m_runner.run();
        }
      }
    };

  private:
    RunnerBase &m_runner;
    std::vector<Question> &m_questions;
  };

  class BareBenchmark {
  public:
    BareBenchmark(RunnerBase &runner, std::vector<Axe> &axes)
        : m_runner(runner),
          m_axes(axes) {};

    std::vector<std::pair<OptionsMap, std::vector<float_milliseconds>>> run() {
      OptionsMap omap;
      m_runner.gather_options(omap);
      std::vector<OptionsMap> omap_v = generate_permutations(omap, m_axes);
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
    std::vector<Axe> &m_axes;
  };

} // namespace Baseliner
#endif // BENCHMARK_HPP