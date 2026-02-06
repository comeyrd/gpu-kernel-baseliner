#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>

auto stop = Baseliner::ConfidenceIntervalMedianSC();
auto runner_matrixmul = []() {
  auto r_ptr = std::make_shared<Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend>>(stop);
  r_ptr->m_block = true;

  return r_ptr;
}();
BASELINER_REGISTER_EXECUTABLE(runner_matrixmul);

Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend> runner_Computation(stop);
BASELINER_REGISTER_EXECUTABLE(&runner_Computation);
