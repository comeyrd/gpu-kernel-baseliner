#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/research_questions/research_questions.hpp>
#include <utility>
#include <vector>
namespace Baseliner {

  class RqBenchmark {
  public:
    RqBenchmark(RunnerBase &runner, std::vector<ResearchQuestions::Question> &questions)
        : m_runner(runner),
          m_questions(questions) {};

    void run() {
      OptionsMap baseMap;
      m_runner.gather_options(baseMap);
      OptionsMap tempMap;
      for (ResearchQuestions::Question &q : m_questions) {
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
    std::vector<ResearchQuestions::Question> &m_questions;
  };

  std::vector<Baseliner::OptionsMap> generate_permutations(Baseliner::OptionsMap base, const std::vector<Axe> &axes,
                                                           int current = 0) {
    const Axe &axe = axes[current];
    current++;
    std::vector<Baseliner::OptionsMap> omaps;
    for (auto value : axe.m_values) {
      Baseliner::OptionsMap inner_om = base;
      inner_om[axe.m_interface_name][axe.m_option_name].m_value = value.m_value;
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