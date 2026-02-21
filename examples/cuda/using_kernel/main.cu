#include "ComputationKernel.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Cuda Kernel Manipuation" << std::endl;

  auto backend = Baseliner::Backend::CudaBackend();
  auto stream = backend.create_stream();
  auto flusher = Baseliner::Backend::L2Flusher<Baseliner::Backend::CudaBackend>();
  auto blocker = Baseliner::Backend::BlockingKernel<Baseliner::Backend::CudaBackend>();
  auto timer = Baseliner::Backend::GpuTimer<Baseliner::Backend::CudaBackend>();
  auto computation_case = Baseliner::KernelCase<ComputationKernel>();
  computation_case.setup(stream);
  computation_case.timed_run(stream);
  std::cout << "Warmup: " << computation_case.time_elapsed().count() << std::endl;

  for (int r = 0; r < 10; r++) {
    computation_case.reset_case(stream);
    flusher.flush(stream);
    blocker.block(stream, 1000.0);
    computation_case.timed_run(stream);
    blocker.unblock();
    std::cout << computation_case.time_elapsed().count() << " | ";
  }
  computation_case.teardown(stream);
  std::cout << std::endl;
}