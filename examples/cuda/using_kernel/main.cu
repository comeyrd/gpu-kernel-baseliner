#include "ComputationKernel.hpp"
#include <iostream>
#include <random>
#include <vector>

int main() {
  std::cout << "Cuda Kernel Manipuation" << std::endl;
  auto input = std::make_shared<ComputationKernel::Input>();
  input->generate_random();
  auto output = ComputationKernel::Output(std::move(input));
  auto impl = ComputationKernel(input);
  auto backend = Baseliner::Backend::CudaBackend();
  auto stream = backend.create_stream();
  auto flusher = Baseliner::Backend::CudaBackend::L2Flusher();
  auto blocker = Baseliner::Backend::CudaBackend::BlockingKernel();
  impl.setup();
  impl.timed_run(stream);
  std::cout << "Warmup: " << impl.time_elapsed().count() << std::endl;

  for (int r = 0; r < 10; r++) {
    flusher.flush(stream);
    blocker.block(stream, 1000.0);
    impl.timed_run(stream);
    blocker.unblock();
    std::cout << impl.time_elapsed().count() << " | ";
  }
  impl.teardown(output);
  std::cout << std::endl;
}