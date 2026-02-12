#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

int main() {
  std::cout << "Cuda Runner Manipuation" << std::endl;
  {
    auto runner_act = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>();
    Baseliner::OptionsMap omap;
    runner_act.gather_options(omap);
    Baseliner::InterfaceOptions &options = omap["Kernel"];
    for (auto &[name, opt] : options) {
      if (name == "work_size") {
        opt.m_value = "8";
      }
    }
    runner_act.propagate_options(omap);
    Baseliner::Result res = runner_act.run();
    serialize(std::cout, res);
    std::cout << std::endl;
  }
  {
    auto runner_act = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>()
                          .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>();
    Baseliner::OptionsMap omap;
    runner_act.gather_options(omap);
    Baseliner::InterfaceOptions &options = omap["Kernel"];
    for (auto &[name, opt] : options) {
      if (name == "work_size") {
        opt.m_value = "8";
      }
    }
    runner_act.propagate_options(omap);
    Baseliner::Result res = runner_act.run();
    serialize(std::cout, res);
    std::cout << std::endl;
  }
}