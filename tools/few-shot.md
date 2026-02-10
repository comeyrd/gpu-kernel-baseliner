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
class IInput : public MoveOnly, public IOptionConsumer {
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
#ifndef MATRIXMUL_KERNEL_HPP
#define MATRIXMUL_KERNEL_HPP

#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <iostream>
#include <random>
#include <vector>

class MatrixMulInput : public Baseliner::IInput {
public:
void register_options() override {
IInput::register_options();
add_option("MatrixMulInput", "wA", "Width of Matrix A", m_wA_base);
add_option("MatrixMulInput", "hA", "Height of Matrix A", m_hA_base);
add_option("MatrixMulInput", "wB", "Width of Matrix B", m_wB_base);
add_option("MatrixMulInput", "hB", "Height of Matrix B", m_hB_base);
add_option("MatrixMulInput", "block_size", "Block size (16 or 32)", m_block_size);
};

void on_update() override {
allocate();
};

void generate_random() override {
std::mt19937 gen(seed);
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto &val : m_h_A)
      val = dist(gen);
    for (auto &val : m_h_B)
      val = dist(gen);

};

explicit MatrixMulInput()
: Baseliner::IInput() {
allocate();
};

void allocate() override {
// Apply work_size multiplier to the height of A to increase workload
m_wA = m_wA_base;
m_hA = m_hA_base \* m_work_size;

    // Inner dimensions must match
    m_hB = m_wA;
    m_wB = m_wB_base;

    m_size_A = m_wA * m_hA;
    m_size_B = m_wB * m_hB;

    m_h_A.resize(m_size_A);
    m_h_B.resize(m_size_B);

}

// Default dimensions based on the original main() example
int m*wA_base = 320; // 5 * 2 _ 32
int m_hA_base = 320;
int m_wB_base = 640; // 5 _ 4 \_ 32
int m_hB_base = 320;

int m_wA, m_hA, m_wB, m_hB;
int m_size_A, m_size_B;
int m_block_size = 32;

std::vector<float> m_h_A;
std::vector<float> m_h_B;
};

class MatrixMulOutput : public Baseliner::IOutput<MatrixMulInput> {
public:
explicit MatrixMulOutput(const MatrixMulInput &input)
: Baseliner::IOutput<MatrixMulInput>(input) {
m_size_C = m_input.m_hA \* m_input.m_wB;
m_h_C.resize(m_size_C);
};

std::vector<float> m_h_C;
int m_size_C;

// Optional: equality check for verification
bool operator==(const MatrixMulOutput &other) const {
if (m_size_C != other.m_size_C)
return false;
for (size_t i = 0; i < m_h_C.size(); i++) {
if (std::abs(m_h_C[i] - other.m_h_C[i]) > 1e-3) {
return false;
}
}
return true;
}
friend std::ostream &operator<<(std::ostream &os, const MatrixMulOutput &thing) {
for (int i = 0; i < thing.m_h_C.size(); i++) {
os << thing.m_h_C[i] << ", ";
}
os << std::endl;
return os;
}
};

class MatrixMulKernel : public Baseliner::ICudaKernel<MatrixMulInput, MatrixMulOutput> {
public:
std::string name() override {
return "MatrixMulKernel";
};
void cpu(MatrixMulOutput &output) override;

void setup() override {
size*t mem_size_A = m_input.m_size_A * sizeof(float);
size*t mem_size_B = m_input.m_size_B * sizeof(float);
size*t mem_size_C = m_input.m_hA * m*input.m_wB * sizeof(float);

    CHECK_CUDA(cudaMalloc(&m_d_A, mem_size_A));
    CHECK_CUDA(cudaMalloc(&m_d_B, mem_size_B));
    CHECK_CUDA(cudaMalloc(&m_d_C, mem_size_C));

    CHECK_CUDA(cudaMemcpy(m_d_A, m_input.m_h_A.data(), mem_size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_d_B, m_input.m_h_B.data(), mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    int block_size = m_input.m_block_size;
    m_threads = dim3(block_size, block_size);
    m_grid = dim3(m_input.m_wB / m_threads.x, m_input.m_hA / m_threads.y);

};

void reset() override {
// Optional: Zero out C if accumulation logic was involved,
// but this kernel writes directly (C = ...), so reset isn't strictly necessary
// unless we want to clear previous results.
size*t mem_size_C = m_input.m_hA * m*input.m_wB * sizeof(float);
CHECK_CUDA(cudaMemset(m_d_C, 0, mem_size_C));
};

void run(std::shared_ptr<cudaStream_t> &stream) override;

void teardown(Output &output) override {
size*t mem_size_C = m_input.m_hA * m*input.m_wB * sizeof(float);
CHECK_CUDA(cudaMemcpy(output.m_h_C.data(), m_d_C, mem_size_C, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(m_d_A));
    CHECK_CUDA(cudaFree(m_d_B));
    CHECK_CUDA(cudaFree(m_d_C));

};

MatrixMulKernel(const MatrixMulInput &input)
: Baseliner::ICudaKernel<Input, Output>(input) {};

private:
float *m_d_A = nullptr;
float *m_d_B = nullptr;
float \*m_d_C = nullptr;
dim3 m_threads;
dim3 m_grid;
};

#endif // MATRIXMUL_KERNEL_HPP

COMPUTATION_KERNEL.CU
#include "MatMul.hpp"

#include <vector>

// ----------------------------------------------------------------------
// Original Kernel Implementation (Unchanged Logic)
// ----------------------------------------------------------------------

/\*\*

- Matrix multiplication (CUDA Kernel) on the device: C = A \* B
- wA is A's width and wB is B's width
  */
  template <int BLOCK_SIZE>
  **global** void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

// Index of the first sub-matrix of A processed by the block
int aBegin = wA _ BLOCK_SIZE _ by;

// Index of the last sub-matrix of A processed by the block
int aEnd = aBegin + wA - 1;

// Step size used to iterate through the sub-matrices of A
int aStep = BLOCK_SIZE;

// Index of the first sub-matrix of B processed by the block
int bBegin = BLOCK_SIZE \* bx;

// Step size used to iterate through the sub-matrices of B
int bStep = BLOCK_SIZE \* wB;

// Csub is used to store the element of the block sub-matrix
// that is computed by the thread
float Csub = 0;

// Loop over all the sub-matrices of A and B
// required to compute the block sub-matrix
for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
// Declaration of the shared memory array As used to
// store the sub-matrix of A
**shared** float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix

#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();

}

// Write the block sub-matrix to device memory;
// each thread writes one element
int c = wB _ BLOCK_SIZE _ by + BLOCK*SIZE * bx;
C[c + wB _ ty + tx] = Csub;
}

// ----------------------------------------------------------------------
// Interface Implementation
// ----------------------------------------------------------------------

void MatrixMulKernel::cpu(MatrixMulOutput &output) {
int wA = m_input.m_wA;
int hA = m_input.m_hA;
int wB = m_input.m_wB;

// Naive CPU implementation for verification
for (int i = 0; i < hA; ++i) {
for (int j = 0; j < wB; ++j) {
double sum = 0;
for (int k = 0; k < wA; ++k) {
double a = m*input.m_h_A[i * wA + k];
double b = m_input.m_h_B[k * wB + j];
sum += a * b;
}
output.m*h_C[i * wB + j] = (float)sum;
}
}
}

void MatrixMulKernel::run(std::shared_ptr<cudaStream_t> &stream) {
// Dispatch based on block size template parameter
if (m_input.m_block_size == 16) {
MatrixMulCUDA<16><<<m_grid, m_threads, 0, *stream>>>(
m_d_C, m_d_A, m_d_B, m_input.m_wA, m_input.m_wB);
} else {
MatrixMulCUDA<32><<<m_grid, m_threads, 0, *stream>>>(
m_d_C, m_d_A, m_d_B, m_input.m_wA, m_input.m_wB);
}
}

---

IF you don't find CPU implementations, simply put a //TODO inside of the CPU impl.

Consider that you need only to find in the code i will give you : Where the device code is (kernels) and put it in the .CU, Where the memory copies are, and put them into setup() and teardown(), put the logic for reseting the memory state in reset(), setup the types for the input and the output.
If the provided kernel does not have a way of creating a random input, take inspiration from the provided generate random to generate random input data with the seed parameter
Do not forget to add the logic to display the output to a stream !
Do not forget the logic to compare two outputs !
Put the run execution parameters not in run, run must only have the kernels instantiation, nothing else
The timing is done around run(), so be carefull
Thanks
Do this with this kernel :
