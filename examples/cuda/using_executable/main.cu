#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>

auto runner_matrixmul = []() {
  auto stop = std::make_unique<Baseliner::ConfidenceIntervalMedianSC>();
  auto r_ptr = std::make_shared<Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend>>(std::move(stop));
  r_ptr->set_block(true);

  return r_ptr;
}();
namespace {
  BASELINER_REGISTER_EXECUTABLE(runner_matrixmul);

  auto stop2 = std::make_unique<Baseliner::ConfidenceIntervalMedianSC>();
  Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_Computation(std::move(stop2));
  BASELINER_REGISTER_EXECUTABLE(&runner_Computation);
} // namespace