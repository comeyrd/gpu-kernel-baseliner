#include "MatMul.hpp"
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>

int main() {
  std::cout << "Cuda Runner Manipuation" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
  Baseliner::Result res = runner_act.run();
  serialize(std::cout, res);
  std::cout << std::endl;
}