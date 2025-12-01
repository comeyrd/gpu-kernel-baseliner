#include "ComputationKernel.hpp"
#include "Runner.hpp"
#include "StoppingCriterion.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Hip Runner Manipuation" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::HipBackend> runner_act(stop);
  std::vector<float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
}
