#include "ComputationKernel.hpp"
#include <baseliner/Durations.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/JsonHandler.hpp>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::cout << "using bare benchmark" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  stop.max_repetitions = 10;
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::HipBackend> runner_act(stop);
  {
  std::vector<Baseliner::Axe> axes = {{"Kernel", "work_size", {"1", "10", "100", "1000"}}, {"Runner", "block", {"0", "1"}}};
  Baseliner::BareBenchmark bare_bench(runner_act,axes);
  auto res = bare_bench.run();
  std::cout << res.size() << std::endl;
  Baseliner::save_to_json(std::cout,res);
  }
  {
  std::vector<Baseliner::Axe> axes = {{"Kernel", "seed", {"1", "10"}}};
  Baseliner::BareBenchmark bare_bench(runner_act,axes);
  auto res = bare_bench.run();
  std::cout << res.size() << std::endl;
  Baseliner::save_to_json(std::cout,res);
  }
}
