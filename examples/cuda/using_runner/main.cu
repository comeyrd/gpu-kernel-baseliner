#include "ComputationKernel.hpp"
#include "Runner.hpp"
#include "StoppingCriterion.hpp"
#include "CpuTimer.hpp"
#include <iostream>
#include <random>
#include <vector>

void simple_runner(){
  std::cout << "Simple Runner " << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_act(stop);
  std::vector<Baseliner::float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
}

void custom_timer_cpu(){
  std::cout <<"Runner with CpuTimer" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  auto cpu_timer = std::make_unique<Baseliner::CpuTimer>();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_act(stop,std::move(cpu_timer));
  std::vector<Baseliner::float_milliseconds> res = runner_act.run();
  std::cout << res << std::endl;
}

int main() {
  std::cout << "Cuda Runner Manipuation" << std::endl;
  simple_runner();
  custom_timer_cpu();
}