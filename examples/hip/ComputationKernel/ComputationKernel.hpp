#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include "Kernel.hpp"
#include "Options.hpp"
#include "backend/hip/HipBackend.hpp"
#include <random>
#include <string>
#include <vector>
constexpr int DEFAULT_N = 5000;

class ComputationInput : public Baseliner::IInput {
public:
  const std::string get_name() override {
    return "ComputationInput";
  }
  void register_options() override {
    IInput::register_options();
    add_option("work_size", "The multiplier of the base work size to apply to the kernel", m_work_size);
    add_option("base_N", "The size of the arrays", m_base_N);
  };
  void on_update() override {
    allocate();
  };
  void generate_random() override {
    std::random_device rd;
    std::mt19937 gen(rd());
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
    m_N = m_base_N * m_work_size;
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
};

class ComputationKernel : public Baseliner::IHipKernel<ComputationInput, ComputationOutput> {
public:
  void cpu(ComputationOutput &output) override;
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
  void run(std::shared_ptr<hipStream_t> &stream) override;
  void teardown(Output &output) override {
    CHECK_HIP(hipMemcpy(output.m_c_host.data(), m_d_c, m_input.m_N * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(m_d_a));
    CHECK_HIP(hipFree(m_d_b));
    CHECK_HIP(hipFree(m_d_c));
  };
  ComputationKernel(const ComputationInput &input)
      : Baseliner::IHipKernel<Input, Output>(input) {};

private:
  dim3 m_threads;
  dim3 m_blocks;
  int *m_d_a;
  int *m_d_b;
  int *m_d_c;
};
#endif // COMPUTATION_HPP
