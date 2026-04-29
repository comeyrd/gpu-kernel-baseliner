#ifndef COMPUTATION_HPP
#define COMPUTATION_HPP

#include <baseliner/core/Workload.hpp>

#include <ostream>
#include <random>
#include <string>
#include <vector>

constexpr int DEFAULT_N = 125000;

template <typename BackendT>
class ComputationWorkload : public Baseliner::IWorkload<BackendT> {
public:
  using backend = typename ComputationWorkload::backend;

  ComputationWorkload() = default;

  // Identification
  auto algo() -> std::string override {
    return "Computation";
  }

  // Metrics
  auto number_of_floating_point_operations() -> std::optional<size_t> override {
    return 151ULL * m_N;
  }
  auto number_of_bytes() -> std::optional<size_t> override {
    return sizeof(int) * m_N + sizeof(int) * m_N;
  }

  // Host setup
  void setup_host_random_generated() override {
    m_N = m_base_N * this->get_work_size();
    m_a_host = std::vector<int>(m_N);
    m_b_host = std::vector<int>(m_N);
    m_c_host = std::vector<int>(m_N);

    std::default_random_engine gen(this->get_seed());
    std::uniform_int_distribution<int> dist(1, 100);
    for (int i = 0; i < m_N; i++) {
      m_a_host[i] = dist(gen);
      m_b_host[i] = dist(gen);
    }
  }
  void setup_host_from_file(std::string &path) override {
    setup_host_random_generated();
    // TODO
  }
  void inner_save_setup(std::string &path) override {};

  void results_from_reference() override;

  // Device lifecycle
  void setup_device(std::shared_ptr<typename backend::stream_t> stream) override;
  void reset_device(std::shared_ptr<typename backend::stream_t> stream) override;
  auto run(std::shared_ptr<typename backend::stream_t> stream) -> typename backend::launch_result_t override;
  void fetch_results(std::shared_ptr<typename backend::stream_t> stream) override;
  void free() override {
    m_a_host.clear();
    m_b_host.clear();
    m_c_host.clear();
    m_c_reference.clear();
  }

  // Validation
  auto validate() -> bool override {
    if (m_c_host.size() != static_cast<size_t>(m_N)) {
      return false;
    }
    const auto &ref = m_c_reference;
    if (ref.size() != static_cast<size_t>(m_N)) {
      return false;
    }
    for (int i = 0; i < m_N; i++) {
      if (m_c_host[i] != ref[i]) {
        return false;
      }
    }
    return true;
  }

protected:
  void register_options() override {
    Baseliner::IWorkload<BackendT>::register_options();
    this->add_option("ComputationWorkload", "base_N", "The size of the arrays", m_base_N);
  }

private:
  // Sizes
  int m_base_N = DEFAULT_N;
  int m_N = 0;

  // Host buffers
  std::vector<int> m_a_host;
  std::vector<int> m_b_host;
  std::vector<int> m_c_host;
  std::vector<int> m_c_reference;

  // Device pointers
  int *m_d_a = nullptr;
  int *m_d_b = nullptr;
  int *m_d_c = nullptr;

  int m_threadsPerBlock;
  int m_blocksPerGrid;
};
template <typename BackendT>
void ComputationWorkload<BackendT>::results_from_reference() {
  m_c_reference = std::vector<int>(m_N);

  for (int idx = 0; idx < m_N; idx++) {
    m_c_reference[idx] = m_a_host[idx] + m_b_host[idx];
  }

  for (int i = 0; i < 50; i++) {
    for (int idx = 1; idx < m_N; idx++) {
      m_c_reference[idx] = m_a_host[idx] + m_c_reference[idx] + m_b_host[idx] * m_b_host[idx];
    }
  }
}

#endif // COMPUTATION_HPP