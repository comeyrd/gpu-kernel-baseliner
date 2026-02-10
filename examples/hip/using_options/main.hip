#include "ComputationKernel.hpp"
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Hip Options Manipuation" << std::endl;
  auto stop = Baseliner::StoppingCriterion();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::HipBackend> runner_act(stop);
  runner_act.m_block = false;
  std::vector<Baseliner::float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
  Baseliner::OptionsMap omap;
  runner_act.gather_options(omap);
  Baseliner::InterfaceOptions &options = omap["ComputationInput"];
  for (auto &[name, opt] : options) {
    if (name == "work_size") {
      opt.m_value = "8";
    }
  }
  runner_act.propagate_options(omap);
  runner_act.m_block = true;
  res = runner_act.run();
  std::cout << res << std::endl;
}
