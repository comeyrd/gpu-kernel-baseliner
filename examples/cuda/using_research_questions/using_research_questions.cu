#include "ComputationKernel.hpp"
#include <baseliner/Benchmark.hpp>
#include <baseliner/Durations.hpp>
#include <baseliner/JsonHandler.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/research_questions/research_questions.hpp>
#include <iostream>

#include <vector>

int main(int argc, char **argv) {
  std::cout << "doing research questions" << std::endl;
  auto stop = Baseliner::StoppingCriterion();
  stop.m_max_repetitions = 10;
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_computation(stop);
  std::vector<Baseliner::ResearchQuestions::Question> research_q = Baseliner::ResearchQuestions::AllRQs;
  Baseliner::RqBenchmark bench(runner_computation, research_q);
  bench.run();
  serialize(std::cout, research_q[0].m_axe.m_values[0].m_results.value());
  std::cout << std::endl;
}
