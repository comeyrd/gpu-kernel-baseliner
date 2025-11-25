#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include "backend/hip/HipBackend.hpp"
#include <random>
#include <vector>
constexpr int DEFAULT_N = 5000;
class Computation : public Baseliner::IHipKernel {
public:
  class Input : public Baseliner::IHipKernel::Input {
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
    explicit Input(const int work_size) : Baseliner::IHipKernel::Input(work_size) {
      m_N = DEFAULT_N * work_size;
      m_a_host = std::vector<int>(m_N);
      m_b_host = std::vector<int>(m_N);
    };
    int m_N;
    std::vector<int> m_a_host;
    std::vector<int> m_b_host;
  };
  class Output : public Baseliner::IHipKernel::Output {
  public:
    void resize(const int work_size) override {
      m_N = DEFAULT_N * work_size;
      m_c_host.resize(m_N);
    };
    explicit Output(const int work_size) : Baseliner::IHipKernel::Output(work_size) {
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
  class GpuImplementation : public Baseliner::IHipKernel::GpuImplementation<Input, Output> {
  public:
    void setup() override {
      CHECK_HIP(hipMalloc(&m_d_a, m_input.m_N * sizeof(int)));
      CHECK_HIP(hipMalloc(&m_d_b, m_input.m_N * sizeof(int)));
      CHECK_HIP(hipMalloc(&m_d_c, m_input.m_N * sizeof(int)));
      int threadsPerBlock = 256;
      int blocksPerGrid = (m_input.m_N + threadsPerBlock - 1) / threadsPerBlock;
      m_threads = dim3(threadsPerBlock);
      m_blocks = dim3(blocksPerGrid);
      CHECK_HIP(hipMemcpy(m_d_a, m_input.m_a_host.data(), m_input.m_N * sizeof(int), hipMemcpyHostToDevice));
      CHECK_HIP(hipMemcpy(m_d_b, m_input.m_b_host.data(), m_input.m_N * sizeof(int), hipMemcpyHostToDevice));
    };
    void reset() override {};
    void run(hipStream_t &stream) override;
    void teardown(Output &output) override {
      CHECK_HIP(hipMemcpy(output.m_c_host.data(), m_d_c, output.m_N * sizeof(int), hipMemcpyDeviceToHost));
      CHECK_HIP(hipFree(m_d_a));
      CHECK_HIP(hipFree(m_d_b));
      CHECK_HIP(hipFree(m_d_c));
  };
  GpuImplementation(const Input &input) : Baseliner::IHipKernel::GpuImplementation<Input, Output>(input) {};

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
