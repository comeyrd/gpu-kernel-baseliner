#include "ComputationKernel.hpp"
#include "Options.hpp"
#include "Runner.hpp"
#include "StoppingCriterion.hpp"
#include <iostream>
#include <random>
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

int main(int argc, char **argv) {
  std::cout << "simple_benchmark" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  stop.max_repetitions = 10;
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
  Baseliner::OptionsMap omap;
  runner_act.gather_options(omap);
  Baseliner::mergeOptionsMap(omap, runner_act.describe_options());
  Baseliner::mergeOptionsMap(omap, stop.describe_options());

  std::vector<Axe> axes = {{"Kernel", "work_size", {"1", "10", "100", "1000"}}, {"Runner", "block", {"0", "1"}}};

  for (Axe &axe : axes) {
    for (std::string value : axe.m_values) {
      omap[axe.m_interface_name][axe.m_option_name].m_value = value;
      runner_act.propagate_options(omap);
      runner_act.apply_options(omap);
      stop.apply_options(omap);
      std::cout << omap << std::endl;
      std::cout << runner_act.run() << std::endl;
    }
  }
  /*
  std::vector<float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
  */
}
