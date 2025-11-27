#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP
#include "Kernel.hpp"
#include "Options.hpp"
#include "backend/cuda/CudaBackend.hpp"
#include <random>
#include <string>
#include <vector>
constexpr int DEFAULT_N = 5000;

class ComputationInput : public Baseliner::IInput {
public:
  const std::string get_name() override{
    return "ComputationInput";
  } 
  std::pair<std::string, Baseliner::InterfaceOptions> describe_options() override {
    Baseliner::InterfaceOptions ComputationInputOptions;
    ComputationInputOptions.push_back({"N", "The number of items in the array for the default work size", std::to_string(DEFAULT_N)});
    return {get_name(), ComputationInputOptions};
  };
  void apply_options(Baseliner::InterfaceOptions &options) override {
    for(auto &option :options){
      if(option.m_name == "N"){
        m_N = std::stoi(option.m_value);
      }
    }
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
  void resize(const int work_size) override {
    m_N = DEFAULT_N * work_size;
    m_a_host.resize(m_N);
    m_b_host.resize(m_N);
  };
  explicit ComputationInput(const int work_size)
      : Baseliner::IInput(work_size) {
    m_N = DEFAULT_N * m_work_size;
    m_a_host = std::vector<int>(m_N);
    m_b_host = std::vector<int>(m_N);
  };
  int m_N;
  std::vector<int> m_a_host;
  std::vector<int> m_b_host;
};

class ComputationOutput : public Baseliner::IOutput {
  public:
    void resize(const int work_size) override {
      m_N = DEFAULT_N * work_size;
      m_c_host.resize(m_N);
    };
    explicit ComputationOutput(const int work_size) : Baseliner::IOutput(work_size) {
      m_N = DEFAULT_N * work_size;
      m_c_host = std::vector<int>(m_N);
    };
    int m_N;
    std::vector<int> m_c_host;
    bool operator==(const ComputationOutput &other) const {
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

class ComputationKernel : public Baseliner::ICudaKernel<ComputationInput,ComputationOutput>{
  public:
    void cpu(ComputationOutput &output) override;
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
    void run(std::shared_ptr<cudaStream_t> &stream) override;
    void teardown(Output &output) override {
      CHECK_CUDA(cudaMemcpy(output.m_c_host.data(), m_d_c, output.m_N * sizeof(int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(m_d_a));
      CHECK_CUDA(cudaFree(m_d_b));
      CHECK_CUDA(cudaFree(m_d_c));
  };
  ComputationKernel(const ComputationInput &input) : Baseliner::ICudaKernel<Input, Output>(input) {};

private:
  dim3 m_threads;
  dim3 m_blocks;
  int *m_d_a;
  int *m_d_b;
  int *m_d_c;
};
#endif // COMPUTATION_HPP
