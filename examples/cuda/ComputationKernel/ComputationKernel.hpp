#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include <baseliner/Kernel.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/backend/cuda/CudaBackend.hpp>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>
constexpr int DEFAULT_N = 5000;

class ComputationInput : public Baseliner::IInput {
public:
  void on_update() override {
    allocate();
  };
  void generate_random() override {
    std::default_random_engine gen(get_seed());
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
    m_N = m_base_N * get_work_size();
    m_a_host = std::vector<int>(m_N);
    m_b_host = std::vector<int>(m_N);
  }

  int m_base_N = DEFAULT_N;
  int m_N = 0;
  std::vector<int> m_a_host;
  std::vector<int> m_b_host;

protected:
  void register_options() override {
    IInput::register_options();
    add_option("ComputationInput", "base_N", "The size of the arrays", m_base_N);
  };
};

class ComputationOutput : public Baseliner::IOutput<ComputationInput> {
public:
  explicit ComputationOutput(std::shared_ptr<const ComputationInput> input)
      : Baseliner::IOutput<ComputationInput>(std::move(input)) {
    m_c_host = std::vector<int>(get_input()->m_N);
  };
  std::vector<int> m_c_host;
  auto operator==(const ComputationOutput &other) const {
    if (get_input()->m_N == other.get_input()->m_N) {
      for (int i = 0; i < get_input()->m_N; i++) {
        if (m_c_host[i] != other.m_c_host[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  friend auto operator<<(std::ostream &oss, const ComputationOutput &thing) -> std::ostream & {
    for (int i = 0; i < thing.get_input()->m_N; i++) {
      oss << thing.m_c_host[i] << ", ";
    }
    oss << std::endl;
    return oss;
  }
};

class ComputationKernel : public Baseliner::ICudaKernel<ComputationInput, ComputationOutput> {
public:
  auto name() -> std::string override {
    return "ComputationKernel";
  };
  void cpu(ComputationOutput &output) override;
  void setup() override {
    CHECK_CUDA(cudaMalloc(&m_d_a, get_input()->m_N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&m_d_b, get_input()->m_N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&m_d_c, get_input()->m_N * sizeof(int)));
    int threadsPerBlock = 256;
    int blocksPerGrid = (get_input()->m_N + threadsPerBlock - 1) / threadsPerBlock;
    m_threads = dim3(threadsPerBlock);
    m_blocks = dim3(blocksPerGrid);
    CHECK_CUDA(cudaMemcpy(m_d_a, get_input()->m_a_host.data(), get_input()->m_N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_d_b, get_input()->m_b_host.data(), get_input()->m_N * sizeof(int), cudaMemcpyHostToDevice));
  };
  void reset() override {};
  void run(std::shared_ptr<cudaStream_t> stream) override;
  void teardown(Output &output) override {
    CHECK_CUDA(cudaMemcpy(output.m_c_host.data(), m_d_c, get_input()->m_N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(m_d_a));
    CHECK_CUDA(cudaFree(m_d_b));
    CHECK_CUDA(cudaFree(m_d_c));
  };
  ComputationKernel(std::shared_ptr<const ComputationInput> input)
      : Baseliner::ICudaKernel<Input, Output>(std::move(input)) {};

private:
  dim3 m_threads;
  dim3 m_blocks;
  int *m_d_a;
  int *m_d_b;
  int *m_d_c;
};
#endif // COMPUTATION_HPP