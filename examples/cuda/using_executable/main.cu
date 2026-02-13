#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
namespace {
  using computation_runner = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>;
  using matrixmul_runner = Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend>;

  static auto runner1 = matrixmul_runner();

  auto runner2 = computation_runner()
                     .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>()
                     .add_stat<Baseliner::Stats::Q1>()
                     .add_stat<Baseliner::Stats::Q3>()
                     .add_stat<Baseliner::Stats::Median>()
                     .add_stat<Baseliner::Stats::WithoutOutliers>()
                     .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();
  BASELINER_REGISTER_EXECUTABLE(&runner1);

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250", "500", "1000", "2000"}};
  Baseliner::SingleAxeBenchmark bench(std::make_shared<computation_runner>(std::move(runner2)), axe);
  BASELINER_REGISTER_EXECUTABLE(&bench);
} // namespace
