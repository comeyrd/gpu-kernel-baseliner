#include <baseliner/Register.hpp>
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/nvbench/NvBenchStopping.hpp>
#include <baseliner/nvbench/StrDelStopping.hpp>
#include <baseliner/primbench/PrimBenchStopping.hpp>
namespace Baseliner {
  // STOPPING CRITERION
  BASELINER_REGISTER_STOPPING_CRITERION(StoppingCriterion);
  BASELINER_REGISTER_STOPPING_CRITERION(ConfidenceIntervalMedianSC);
  BASELINER_REGISTER_STOPPING_CRITERION(VariationStoppingCriterion);
  BASELINER_REGISTER_STOPPING_CRITERION(EntropyStoppingCriterion);
  BASELINER_REGISTER_STOPPING_CRITERION(StdRelStoppingCriterion);

  // STATS
  namespace Stats {
    BASELINER_REGISTER_STAT(SortedExecutionTimeVector);
    BASELINER_REGISTER_STAT(Median);
    BASELINER_REGISTER_STAT(DataThroughput);
    BASELINER_REGISTER_STAT(FLOPThroughput);
    BASELINER_REGISTER_STAT(Q1);
    BASELINER_REGISTER_STAT(Q3);
    BASELINER_REGISTER_STAT(MedianConfidenceInterval);
    BASELINER_REGISTER_STAT(WithoutOutliers);
    BASELINER_REGISTER_STAT(MedianAbsoluteDeviation);
    BASELINER_REGISTER_STAT(Mean);
    BASELINER_REGISTER_STAT(HarmonicMean);
    const std::vector<std::string> default_stats = {"Median", "Mean", "CoefficientOfVariation"};
    BASELINER_REGISTER_DEFAULT_STATS(default_stats);
  } // namespace Stats

} // namespace Baseliner