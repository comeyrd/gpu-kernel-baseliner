#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include "backend/cuda/CudaBackend.hpp"
#include <random>
#include <vector>
constexpr int DEFAULT_N = 5000;
class Computation : public Baseliner::ICudaKernel {
public:
  class Input : public Baseliner::ICudaKernel::Input {
  public:
    void generate_random() override {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> dist(1, 100);
      for (int i = 0; i < m_N; i++) {
        m_a_host[i] = dist(gen);
        m_b_host[i] = dist(gen);
      }
    };
    void resize(const int work_size) override {
      m_N = DEFAULT_N * work_size;
      m_a_host.resize(m_N);
      m_b_host.resize(m_N);
    };
    explicit Input(const int work_size) : Baseliner::ICudaKernel::Input(work_size) {
      m_N = DEFAULT_N * work_size;
      m_a_host = std::vector<int>(m_N);
      m_b_host = std::vector<int>(m_N);
    };
    int m_N;
    std::vector<int> m_a_host;
    std::vector<int> m_b_host;
  };
  class Output : public Baseliner::ICudaKernel::Output {
  public:
    void resize(const int work_size) override {
      m_N = DEFAULT_N * work_size;
      m_c_host.resize(m_N);
    };
    explicit Output(const int work_size) : Baseliner::ICudaKernel::Output(work_size) {
      m_N = DEFAULT_N * work_size;
      m_c_host = std::vector<int>(m_N);
    };
    int m_N;
    std::vector<int> m_c_host;
    bool operator==(const Output &other) const {
      if (m_N == other.m_N) {
        for (int i = 0; i < m_N; i++) {
          if (m_c_host[i] != other.m_c_host[i]) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
  };
  class GpuImplementation : public Baseliner::ICudaKernel::GpuImplementation<Input, Output> {
  public:
    void setup() override {
      CHECK_CUDA(cudaMalloc(&m_d_a, m_input.m_N * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&m_d_b, m_input.m_N * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&m_d_c, m_input.m_N * sizeof(int)));
      int threadsPerBlock = 256;
      int blocksPerGrid = (m_input.m_N + threadsPerBlock - 1) / threadsPerBlock;
      m_threads = dim3(threadsPerBlock);
      m_blocks = dim3(blocksPerGrid);
      CHECK_CUDA(cudaMemcpy(m_d_a, m_input.m_a_host.data(), m_input.m_N * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(m_d_b, m_input.m_b_host.data(), m_input.m_N * sizeof(int), cudaMemcpyHostToDevice));
    };
    void reset() override {};
    void run(cudaStream_t &stream) override;
    void teardown(Output &output) override {
      CHECK_CUDA(cudaMemcpy(output.m_c_host.data(), m_d_c, output.m_N * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(m_d_a));
      CHECK_CUDA(cudaFree(m_d_b));
      CHECK_CUDA(cudaFree(m_d_c));
  };
  GpuImplementation(const Input &input) : Baseliner::ICudaKernel::GpuImplementation<Input, Output>(input) {};

private:
  dim3 m_threads;
  dim3 m_blocks;
  int *m_d_a;
  int *m_d_b;
  int *m_d_c;
};
}
;

#endif // COMPUTATION_HPP
