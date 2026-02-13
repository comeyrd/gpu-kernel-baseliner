#include "ComputationKernel.hpp"
#include "MatMul.hpp"
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
  BASELINER_REGISTER_EXECUTABLE(&runner2);
} // namespace
