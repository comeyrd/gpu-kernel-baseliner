#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

int main() {
  std::cout << "Cuda Runner Manipuation" << std::endl;
  {
    auto benchmark_act = Baseliner::CudaBenchmark().set_kernel<ComputationKernel>();
    Baseliner::OptionsMap omap;
    benchmark_act.gather_options(omap);
    Baseliner::InterfaceOptions &options = omap["Kernel"];
    for (auto &[name, opt] : options) {
      if (name == "work_size") {
        opt.m_value = "8";
      }
    }
    benchmark_act.propagate_options(omap);
    benchmark_act.gather_options();
    Baseliner::Result res = benchmark_act.run();
    serialize(std::cout, res);
    std::cout << std::endl;
  }
  {
    static auto benchmark_act = Baseliner::CudaBenchmark()
                                    .set_kernel<ComputationKernel>()
                                    .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>();
    Baseliner::OptionsMap omap;
    benchmark_act.gather_options(omap);
    Baseliner::InterfaceOptions &options = omap["Kernel"];
    for (auto &[name, opt] : options) {
      if (name == "work_size") {
        opt.m_value = "8";
      }
    }
    benchmark_act.propagate_options(omap);
    Baseliner::Result res = benchmark_act.run();
    serialize(std::cout, res);
    std::cout << std::endl;
  }
}