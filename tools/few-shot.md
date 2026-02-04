I need you to write code to port a CUDA kernel to a specific interface
DO not make any change to the kernel itself or its logic.
Just move code around to fit this specific interface.
Every cuda call MUST be protected with CHECK_CUDA() around it
This is the base interface that you will base your code on :

KERNEL.HPP

---

#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <baseliner/Options.hpp>
#include <memory>
namespace Baseliner {

// Move only semantics base class
class MoveOnly {
protected:
MoveOnly() = default;

public:
MoveOnly(const MoveOnly &) = delete;
MoveOnly &operator=(const MoveOnly &) = delete;
MoveOnly(MoveOnly &&) noexcept = default;
MoveOnly &operator=(MoveOnly &&) noexcept = default;
virtual ~MoveOnly() = default;
};
// TODO Work on the instantiation of InputData : Reusing old data and saving data to file.
class IInput : public MoveOnly, public OptionConsumer {
public:
void register_options() override {
add_option("Kernel", "work_size", "The multiplier of the base work size to apply to the kernel", m_work_size);
add_option("Kernel", "seed", "The seed used for the generation of input data", seed);
}
virtual void generate_random() = 0;

protected:
virtual void allocate() = 0;
int m_work_size = 1;
int seed = 202;
IInput() = default;
virtual ~IInput() = default;
};
template <typename Input>
class IOutput : public MoveOnly {
public:
protected:
const Input &m_input;
IOutput(const Input &input)
: m_input(input) {};
virtual ~IOutput() = default;
};

template <typename stream_t, typename I, typename O>
class IKernel {
public:
using Input = I;
using Output = O;
virtual void cpu(Output &output) = 0;
virtual void setup() = 0;
virtual void reset() = 0;
virtual void run(std::shared_ptr<stream_t> &stream) = 0;
virtual void teardown(Output &output) = 0;
IKernel(const Input &input)
: m_input(input) {};
virtual ~IKernel() = default;

protected:
const Input &m_input;
};
} // namespace Baseliner

#endif // KERNEL_HPP

---

This is an example of Cuda Kernel ported to this architecture :

COMPUTATION_KERNEL.HPP
#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <random>
#include <string>
#include <vector>
constexpr int DEFAULT_N = 5000;

class ComputationInput : public Baseliner::IInput {
public:
void register_options() override {
IInput::register_options();
add_option("ComputationInput", "base_N", "The size of the arrays", m_base_N);
};
void on_update() override {
allocate();
};
void generate_random() override {
std::default_random_engine gen(seed);
std::uniform_int_distribution<int> dist(1, 100);
for (int i = 0; i < m_N; i++) {
m_a_host[i] = dist(gen);
m_b_host[i] = dist(gen);
}
};
explicit ComputationInput()
: Baseliner::IInput() {
allocate();
};
void allocate() override {
m_N = m_base_N \* m_work_size;
m_a_host = std::vector<int>(m_N);
m_b_host = std::vector<int>(m_N);
}
int m_base_N = DEFAULT_N;
int m_N;
std::vector<int> m_a_host;
std::vector<int> m_b_host;
};

class ComputationOutput : public Baseliner::IOutput<ComputationInput> {
public:
explicit ComputationOutput(const ComputationInput &input)
: Baseliner::IOutput<ComputationInput>(input) {
m_c_host = std::vector<int>(m_input.m_N);
};
std::vector<int> m_c_host;
bool operator==(const ComputationOutput &other) const {
if (m_input.m_N == other.m_input.m_N) {
for (int i = 0; i < m_input.m_N; i++) {
if (m_c_host[i] != other.m_c_host[i]) {
return false;
}
}
return true;
}
return false;
}
friend std::ostream &operator<<(std::ostream &os, const ComputationOutput &thing) {
for (int i = 0; i < thing.m_input.m_N; i++) {
os << thing.m_c_host[i] << ", ";
}
os << std::endl;
return os;
}
};

class ComputationKernel : public Baseliner::ICudaKernel<ComputationInput, ComputationOutput> {
public:
void cpu(ComputationOutput &output) override;
void setup() override {
CHECK*CUDA(cudaMalloc(&m_d_a, m_input.m_N * sizeof(int)));
CHECK*CUDA(cudaMalloc(&m_d_b, m_input.m_N * sizeof(int)));
CHECK*CUDA(cudaMalloc(&m_d_c, m_input.m_N * sizeof(int)));
int threadsPerBlock = 256;
int blocksPerGrid = (m*input.m_N + threadsPerBlock - 1) / threadsPerBlock;
m_threads = dim3(threadsPerBlock);
m_blocks = dim3(blocksPerGrid);
CHECK_CUDA(cudaMemcpy(m_d_a, m_input.m_a_host.data(), m_input.m_N * sizeof(int), cudaMemcpyHostToDevice));
CHECK*CUDA(cudaMemcpy(m_d_b, m_input.m_b_host.data(), m_input.m_N * sizeof(int), cudaMemcpyHostToDevice));
};
void reset() override {};
void run(std::shared*ptr<cudaStream_t> &stream) override;
void teardown(Output &output) override {
CHECK_CUDA(cudaMemcpy(output.m_c_host.data(), m_d_c, m_input.m_N * sizeof(int), cudaMemcpyDeviceToHost));
CHECK_CUDA(cudaFree(m_d_a));
CHECK_CUDA(cudaFree(m_d_b));
CHECK_CUDA(cudaFree(m_d_c));
};
ComputationKernel(const ComputationInput &input)
: Baseliner::ICudaKernel<Input, Output>(input) {};

private:
dim3 m_threads;
dim3 m_blocks;
int *m_d_a;
int *m_d_b;
int \*m_d_c;
};
#endif // COMPUTATION_HPP

COMPUTATION_KERNEL.CU
#include "ComputationKernel.hpp"
#include <iostream>
#include <random>
#include <vector>

**global** void computation*kernel(int *a, int _b, int \_c, int N) {
int idx = blockIdx.x _ blockDim.x + threadIdx.x;

if (idx < N) {
c[idx] = a[idx] + b[idx];
}
**syncthreads();
for (int i = 0; i < 50; i++) {
if (idx > 0 && idx < N) {
c[idx] = a[idx] + c[idx] + b[idx] \* b[idx];
}
**syncthreads();
}
}

void ComputationKernel::cpu(ComputationOutput &output) {
for (int i = 0; i < m*input.m_N; ++i) {
output.m_c_host[i] = m_input.m_a_host[i] + m_input.m_b_host[i];
}
for (int * = 0; _ < 50; _++) {
for (int i = 1; i < m_input.m_N; ++i) {
output.m_c_host[i] = m_input.m_a_host[i] + output.m_c_host[i] + (m_input.m_b_host[i] \* m_input.m_b_host[i]);
}
}
}

void ComputationKernel::run(std::shared_ptr<cudaStream_t> &stream) {
computation_kernel<<<m_blocks, m_threads, 0, \*stream>>>(m_d_a, m_d_b, m_d_c, m_input.m_N);
}

---

IF you don't find CPU implementations, simply put a //TODO inside of the CPU impl.

Consider that you need only to find in the code i will give you : Where the device code is (kernels) and put it in the .CU, Where the memory copies are, and put them into setup() and teardown(), put the logic for reseting the memory state in reset(), setup the types for the input and the output.
If the provided kernel does not have a way of creating a random input, take inspiration from the provided generate random to generate random input data with the seed parameter
Do not forget to add the logic to display the output to a stream !
Thanks
Do this with this kernel :
