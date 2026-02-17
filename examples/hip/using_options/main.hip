#include "ComputationKernel.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

int main() {
  std::cout << "Hip Options Manipuation" << std::endl;
  auto benchmark_act = Baseliner::HipBenchmark().set_kernel<ComputationKernel>();
  benchmark_act.set_block(false);
  Baseliner::Result res = benchmark_act.run();
  serialize(std::cout, res);
  std::cout << std::endl;

  Baseliner::OptionsMap omap;
  benchmark_act.gather_options(omap);
  Baseliner::InterfaceOptions &options = omap["ComputationInput"];
  for (auto &[name, opt] : options) {
    if (name == "work_size") {
      opt.m_value = "8";
    }
  }
  benchmark_act.propagate_options(omap);
  benchmark_act.set_block(true);
  Baseliner::Result second_res = benchmark_act.run();
  serialize(std::cout, second_res);

  std::cout << std::endl;
}
