#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/registry/RegisteringMacros.hpp>
namespace Baseliner {
  BASELINER_REGISTER_STOPPING_CRITERION(StoppingCriterion);
  BASELINER_REGISTER_STOPPING_CRITERION(ConfidenceIntervalMedianSC);
  BASELINER_REGISTER_STOPPING_CRITERION(VariationStoppingCriterion);
} // namespace Baseliner