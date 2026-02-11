#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>
namespace {
  using computation_runner = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>;
  using matrixmul_runner = Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend>;

  // Store by VALUE, not reference
  static auto runner1 = matrixmul_runner();

  // Create a lambda or a helper to initialize and configure in one go
  namespace {
    static auto runner2 = []() {
      auto r = std::make_unique<computation_runner>();
      r->set_stopping_criterion<Baseliner::ConfidenceIntervalMedianSC>();
      return r;
    }();

    // Note: You must pass the raw pointer to the register macro
    BASELINER_REGISTER_EXECUTABLE(&runner1);
    BASELINER_REGISTER_EXECUTABLE(runner2.get());
  } // namespace
} // namespace