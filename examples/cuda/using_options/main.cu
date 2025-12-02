#include "ComputationKernel.hpp"
#include "Runner.hpp"
#include "StoppingCriterion.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Cuda Options Manipuation" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
  runner_act.m_block = false;
  std::vector<float_milliseconds> res = runner_act.run();
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