#include "ComputationKernel.hpp"
#include "Durations.hpp"
#include "Options.hpp"
#include "Runner.hpp"
#include "StoppingCriterion.hpp"
#include "research_questions/research_questions.hpp"
#include "Benchmark.hpp"
#include "JsonHandler.hpp"
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::cout << "doing research questions" << std::endl;
  auto stop = Baseliner::FixedRepetitionStoppingCriterion();
  stop.max_repetitions = 10;
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_computation(stop);
  std::vector<Basliner::Question> research_q = Baseliner::ResearchQuestions::AllRQs;
  Baseliner::RqBenchmark bench(runner_computation,research_q);
  bench.run();
  std::cout << research_q[0].m_axe.m_values[0].m_results.size() <<std::endl;

}
