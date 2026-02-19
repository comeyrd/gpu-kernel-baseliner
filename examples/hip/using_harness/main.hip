#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Axe.hpp>
#include <baseliner/Benchmark.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/Suite.hpp>
#include <baseliner/Task.hpp>
#include <baseliner/backend/BackendStats.hpp>
namespace {
  static auto benchmark1 = Baseliner::HipBenchmark().set_kernel<ComputationKernel>();

  auto benchmark2 = Baseliner::HipBenchmark()
                        .set_kernel<MatrixMulKernel>()
                        .add_stat<Baseliner::Stats::Q1>()
                        .add_stat<Baseliner::Stats::Q3>()
                        .add_stat<Baseliner::Stats::Median>()
                        .add_stat<Baseliner::Stats::WithoutOutliers>()
                        .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>()                        
                        .add_stat<Baseliner::Stats::ClockFrequencyVector<Baseliner::Device::HipBackend>>();


  BASELINER_REGISTER_TASK(&benchmark1);

  Baseliner::Axe axe = {"StoppingCriterion", "max_nb_repetition", {"100", "250","500","750","1000","2000"}};
  Baseliner::SingleAxeSuite bench(std::make_shared<Baseliner::HipBenchmark>(std::move(benchmark2)), std::move(axe));
  BASELINER_REGISTER_TASK(&bench);
} // namespace

namespace ase{

  auto benchmark2 = Baseliner::HipBenchmark()
                        .set_kernel<MatrixMulKernel>()
                        .add_stat<Baseliner::Stats::Q1>()
                        .add_stat<Baseliner::Stats::Q3>()
                        .add_stat<Baseliner::Stats::Median>()
                        .add_stat<Baseliner::Stats::WithoutOutliers>()
                        .add_stat<Baseliner::Stats::MedianAbsoluteDeviation>()                        
                        .add_stat<Baseliner::Stats::ClockFrequencyVector<Baseliner::Device::HipBackend>>()
                        .add_stat<Baseliner::Stats::MedianItemTroughput>();


  BASELINER_REGISTER_TASK(&benchmark1);

  Baseliner::Axe axe = {"Kernel", "work_size", {"1", "2","4","8","16","32"}};
  Baseliner::SingleAxeSuite bench(std::make_shared<Baseliner::HipBenchmark>(std::move(benchmark2)), std::move(axe));
  BASELINER_REGISTER_TASK(&bench);
}