#include "MatMul.hpp"
#include <baseliner/Runner.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Cuda Runner Manipuation" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
  std::vector<Baseliner::float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
}