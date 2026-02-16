#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Task.hpp>
namespace {
  static auto benchmark1 = Baseliner::CudaBenchmark().set_kernel<ComputationKernel>();

  auto benchmark2 = Baseliner::CudaBenchmark()
                        .set_kernel<MatrixMulKernel>()
                        .set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>()
                        .add_stat<Baseliner::Stats::Q1>()
                        .add_stat<Baseliner::Stats::Q3>()
                        .add_stat<Baseliner::Stats::Median>()
                        .add_stat<Baseliner::Stats::WithoutOutliers>()
                        .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>();
  //.add_stat<Baseliner::Stats::SnEstimator>()
  //.add_stat<Baseliner::Stats::QnEstimator>();

  BASELINER_REGISTER_TASK(&benchmark1);

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250", "500", "1000", "2000"}};
  Baseliner::SingleAxeSuite bench(std::make_shared<Baseliner::CudaBenchmark>(std::move(benchmark2)), std::move(axe));
  BASELINER_REGISTER_TASK(&bench);
} // namespace
