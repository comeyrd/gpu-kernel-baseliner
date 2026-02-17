#include "ComputationKernel.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>
#include <string>
#include <vector>

inline std::ostream &operator<<(std::ostream &os, const Baseliner::OptionsMap &option_map) {
  os << "{" << std::endl;
  for (const auto &[key, val] : option_map) {
    os << "  " << key << " : {" << std::endl;
    for (auto [name, opt] : val) {
      os << "    " << name << " : " << opt.m_value << " ," << std::endl;
    }
    os << "  }," << std::endl;
  }
  os << "}" << std::endl;
  return os;
}
struct Axe {
  std::string m_interface_name;
  std::string m_option_name;
  std::vector<std::string> m_values;
};
std::vector<Baseliner::OptionsMap> generate_permutations(Baseliner::OptionsMap base, const std::vector<Axe> &axes,
                                                         int current = 0) {
  const Axe &axe = axes[current];
  current++;
  std::vector<Baseliner::OptionsMap> omaps;
  for (std::string value : axe.m_values) {
    Baseliner::OptionsMap inner_om = base;
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

class IExploration {
public:
  virtual bool done() = 0;
  virtual void applyResults(Baseliner::Result results) = 0;
  virtual Baseliner::OptionsMap next() = 0;
};

using Score = float;

static Score grade(Baseliner::Result res) {
  Score mean = 0.0;

  size_t size = 1;
  for (auto metric : res.get_v_metrics()) {
    if (metric.m_name == "execution_time") {
      auto *vec_ptr = std::get_if<std::vector<Baseliner::float_milliseconds>>(&metric.m_data);
      if (vec_ptr) {
        size = vec_ptr->size();
        for (const auto &duration : *vec_ptr) {
          mean += duration.count();
        }
      }
    }
  }
  mean = mean / size;
  return mean;
}
class AxeExploration : public IExploration {
public:
  bool done() override {
    if (m_current < m_options_maps.size()) {
      return false;
    } else {
      return true;
    }
  };
  void applyResults(Baseliner::Result results) override {
    m_scores.push_back(grade(results));
  };
  Baseliner::OptionsMap next() override {
    Baseliner::OptionsMap nextomap = m_options_maps[m_current];
    m_current++;
    return nextomap;
  };
  AxeExploration(std::vector<Axe> axes, Baseliner::OptionsMap &baseOptionsMap) {
    m_options_maps = generate_permutations(baseOptionsMap, axes);
  }

  std::vector<Baseliner::OptionsMap> m_options_maps;
  std::vector<Score> m_scores;

private:
  size_t m_current = 0;
};

class Benchmark {
public:
  void run(Baseliner::IBenchmark &benchmark, IExploration &exploration) {
    while (!exploration.done()) {
      Baseliner::OptionsMap options = exploration.next();
      benchmark.propagate_options(options);
      Baseliner::Result results = benchmark.run();
      exploration.applyResults(results);
    }
  };
};

int main(int argc, char **argv) {
  std::cout << "simple_benchmark" << std::endl;
  static auto benchmark1 =
      Baseliner::HipBenchmark().set_kernel<ComputationKernel>().set_stopping_criterion<Baseliner::StoppingCriterion>(
          25, 1);
  ;
  Baseliner::OptionsMap omap;
  std::vector<Axe> axes = {{"Kernel", "work_size", {"1", "10", "100", "1000"}}, {"Runner", "block", {"0", "1"}}};
  AxeExploration axe_exp = AxeExploration(axes, omap);
  Benchmark bench;
  bench.run(benchmark1, axe_exp);
  for (int i = 0; i < axe_exp.m_options_maps.size(); i++) {
    std::cout << axe_exp.m_options_maps[i] << std::endl;
    std::cout << "Score : " << axe_exp.m_scores[i] << std::endl;
    std::cout << "-------------" << std::endl;
  }
  /*
  std::vector<float_milliseconds> res = benchmark_act.run();
  std::cout << res << std::endl;
  */
}
