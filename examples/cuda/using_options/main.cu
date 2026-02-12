#include "ComputationKernel.hpp"
#include <baseliner/Result.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

int main() {
  std::cout << "Cuda Options Manipuation" << std::endl;
  auto runner_act = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>();
  runner_act.set_block(false);
  Baseliner::Result res = runner_act.run();
  serialize(std::cout, res);
  std::cout << std::endl;

  Baseliner::OptionsMap omap;
  runner_act.gather_options(omap);
  Baseliner::InterfaceOptions &options = omap["ComputationInput"];
  for (auto &[name, opt] : options) {
    if (name == "work_size") {
      opt.m_value = "8";
    }
  }
  runner_act.propagate_options(omap);
  runner_act.set_block(true);
  Baseliner::Result second_res = runner_act.run();
  serialize(std::cout, second_res);

  std::cout << std::endl;
}