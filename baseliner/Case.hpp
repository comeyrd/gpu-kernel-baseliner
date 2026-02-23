#ifndef BASELINER_CASE_HPP
#define BASELINER_CASE_HPP
#include <baseliner/Options.hpp>
#include <baseliner/Task.hpp>
#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>
#include <baseliner/stats/Stats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <memory>
#include <optional>
namespace Baseliner {
  template <typename BackendT>
  class ICase : public Backend::GpuTimer<BackendT>, public LazyOption {
  public:
    using backend = BackendT;
    using Backend::GpuTimer<BackendT>::time_elapsed;
    ICase() = default;
    ~ICase() override = default;
    virtual auto name() -> std::string = 0;
    virtual void setup(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void run_case(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual void teardown(std::shared_ptr<typename BackendT::stream_t> stream) = 0;
    virtual auto validate_case() -> bool = 0;

    virtual void timed_run(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      run_case(stream);
      this->measure_stop(stream);
    };
    virtual void time_setup(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      setup(stream);
      this->measure_stop(stream);
    };
    virtual void time_teardown(std::shared_ptr<typename BackendT::stream_t> stream) {
      this->measure_start(stream);
      teardown(stream);
      this->measure_stop(stream);
    };
  };

  template <typename Kernel>
  class KernelCase : public ICase<typename Kernel::backend> {
    using BackendT = typename Kernel::backend;

  public:
    KernelCase()
        : m_input(std::make_shared<typename Kernel::Input>()),
          m_kernel(std::make_unique<Kernel>(m_input)) {};
    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_input->generate_random();
      m_kernel->setup(stream);
    };
    void reset_case(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_kernel->reset_kernel(stream);
    }
    void run_case(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_kernel->run(stream);
    }
    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_gpu_output = std::make_shared<typename Kernel::Output>(m_input);
      m_kernel->teardown(stream, *m_gpu_output);
    }
    auto validate_case() -> bool override {
      if (m_comparison_output && m_gpu_output) {
        return *m_gpu_output == *m_comparison_output;
      }
      return true;
    }
    void register_options_dependencies() override {
      this->register_consumer(*m_input);
    }
    auto name() -> std::string override {
      return m_kernel->name();
    }
    void setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      std::optional<size_t> bytes = m_kernel->number_of_bytes();
      std::optional<size_t> flops = m_kernel->number_of_floating_point_operations();
      if (bytes.has_value()) {
        engine->register_metric<Stats::ByteNumbers>(bytes.value());
        m_bytes = true;
      }
      if (flops.has_value()) {
        engine->register_metric<Stats::FLOPCount>(flops.value());
        m_bytes = true;
      }
      m_kernel->setup_metrics(engine);
    };
    void update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      std::optional<size_t> bytes = m_kernel->number_of_bytes();
      std::optional<size_t> flops = m_kernel->number_of_floating_point_operations();
      if (m_bytes) {
        engine->update_values<Stats::ByteNumbers>(bytes.value());
      }
      if (m_flops) {
        engine->update_values<Stats::FLOPCount>(flops.value());
      }
      m_kernel->update_metrics(engine);
    };

  private:
    bool m_flops = false;
    bool m_bytes = false;
    std::shared_ptr<typename Kernel::Input> m_input;
    std::unique_ptr<Kernel> m_kernel;
    std::shared_ptr<typename Kernel::Output> m_gpu_output;
    std::shared_ptr<typename Kernel::Output> m_comparison_output;
  };

} // namespace Baseliner
#endif // BASELINER_CASE_HPP