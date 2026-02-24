#include <baseliner/Axe.hpp>
#include <baseliner/managers/BenchmarkCaseManager.hpp>
#include <baseliner/managers/SuiteManager.hpp>
#include <string>
#include <vector>

namespace Baseliner::ResearchQuestions {

  const Axe rq1 = {
      "RQ1", "Does different input values impact kernel execution times ?", "Kernel", "seed", {"123", "4444", "2133"}};
  BASELINER_REGISTER_SUITE(rq1);

  const Axe rq2 = {"RQ2",
                   "What impact has the work size on the kernel execution time ?",
                   "Kernel",
                   "work_size",
                   {"1", "2", "3", "4", "5", "6", "8", "9", "10", "12", "14", "16"}};
  BASELINER_REGISTER_SUITE(rq2);

  const Axe rq3 = {
      "RQ3", "How does flushing the L2 cache impact the kernel execution time", "Benchmark", "flush", {"0", "1"}};
  BASELINER_REGISTER_SUITE(rq3);

  const Axe rq4 = {"RQ4",
                   "What impact has the enqueing or not of kernels on it's execution time ?",
                   "Benchmark",
                   "block",
                   {"0", "1"}};
  BASELINER_REGISTER_SUITE(rq4);

  const Axe rq5 = {"RQ5", "How does warmups impact the kernel execution time ?", "Benchmark", "warmup", {"0", "1"}};
  BASELINER_REGISTER_SUITE(rq5);

} // namespace Baseliner::ResearchQuestions
