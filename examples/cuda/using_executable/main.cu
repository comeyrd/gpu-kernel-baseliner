#include "ComputationKernel.hpp"
#include "MatMul.hpp"
#include <baseliner/Executable.hpp>
#include <baseliner/Runner.hpp>
#include <baseliner/Serializer.hpp>
#include <baseliner/StoppingCriterion.hpp>

namespace {

  using computation_runner = Baseliner::Runner<ComputationKernel, Baseliner::Backend::CudaBackend>;
  using matrixmul_runner = Baseliner::Runner<MatrixMulKernel, Baseliner::Backend::CudaBackend>;

  auto runner1 = matrixmul_runner();
  auto runner2 = computation_runner();

  BASELINER_REGISTER_EXECUTABLE(&runner1);
  BASELINER_REGISTER_EXECUTABLE(&runner2);

} // namespace