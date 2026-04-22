#ifndef BASELINER_CORE_KERNEL_HPP
#define BASELINER_CORE_KERNEL_HPP
#include <baseliner/core/Options.hpp>
#include <baseliner/core/Timer.hpp>
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/core/stats/StatsEngine.hpp>
#include <memory>
#include <optional>
namespace Baseliner {

  // Move only semantics base class
  class MoveOnly {
  protected:
    MoveOnly() = default;

  public:
    MoveOnly(const MoveOnly &) = delete;
    auto operator=(const MoveOnly &) -> MoveOnly & = delete;
    MoveOnly(MoveOnly &&) noexcept = default;
    auto operator=(MoveOnly &&) noexcept -> MoveOnly & = default;
    virtual ~MoveOnly() = default;
  };
  class IInput : public MoveOnly, public IOption {
  public:
    virtual void generate_random(int seed) = 0;
    virtual void allocate(int work_size) = 0;

    ~IInput() override = default;
    IInput() = default;
    virtual auto number_of_floating_point_operations() -> std::optional<size_t> {
      return {};
    }
    virtual auto number_of_bytes() -> std::optional<size_t> {
      return {};
    }
  };
  template <typename Input>
  class IOutput : public MoveOnly {
  public:
    auto get_input() const -> std::shared_ptr<const Input> {
      return m_input;
    };

    IOutput(std::shared_ptr<const Input> input)
        : m_input(input) {};
    ~IOutput() override = default;

  private:
    std::shared_ptr<const Input> m_input;
  };

  template <typename B, typename I, typename O>
  class IKernel {
  public:
    using Input = I;
    using Output = O;
    using backend = B;
    virtual void setup(std::shared_ptr<typename backend::stream_t> stream) = 0;
    virtual void reset_kernel(std::shared_ptr<typename backend::stream_t> stream) = 0;
    virtual void setup_metrics(std::shared_ptr<Stats::StatsEngine> & /*engine*/) {};  // NOLINT
    virtual void update_metrics(std::shared_ptr<Stats::StatsEngine> & /*engine*/) {}; // NOLINT
    virtual auto run(std::shared_ptr<typename backend::stream_t> stream) -> typename backend::launch_result_t = 0;
    virtual void teardown(std::shared_ptr<typename backend::stream_t> stream, Output &output) = 0;
    virtual auto name() -> std::string = 0;
    IKernel(const std::shared_ptr<const Input> input)
        : m_input(input) {};
    virtual ~IKernel() = default;

    auto get_input() -> std::shared_ptr<const Input> {
      return m_input;
    };

  private:
    std::shared_ptr<const Input> m_input;
  };

  template <typename Kernel>
  class KernelWorkload : public IWorkload<typename Kernel::backend> {
    using BackendT = typename Kernel::backend;

  public:
    KernelWorkload()
        : m_input(std::make_shared<typename Kernel::Input>()),
          m_kernel(std::make_unique<Kernel>(m_input)) {};
    void setup(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_input->allocate(this->get_work_size());
      m_input->generate_random(this->get_seed());
      m_kernel->setup(stream);
    };
    void reset_workload(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_kernel->reset_kernel(stream);
    }
    auto run_workload(std::shared_ptr<typename BackendT::stream_t> stream) -> std::monostate override {
      return m_kernel->run(stream);
    }
    void teardown(std::shared_ptr<typename BackendT::stream_t> stream) override {
      m_gpu_output = std::make_shared<typename Kernel::Output>(m_input);
      m_kernel->teardown(stream, *m_gpu_output);
    }
    void workload_setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      m_kernel->setup_metrics(engine);
    };
    void workload_update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      m_kernel->update_metrics(engine);
    };
    auto validate_workload() -> bool override {
      if (m_comparison_output && m_gpu_output) {
        return *m_gpu_output == *m_comparison_output;
      }
      return true;
    }
    void register_options_dependencies() override {
      this->register_consumer(m_input.get());
    }
    auto name() -> std::string override {
      return m_kernel->name();
    }

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return m_input->number_of_floating_point_operations();
    }
    auto number_of_bytes() -> std::optional<size_t> override {
      return m_input->number_of_bytes();
    }

    void on_update() override {
      IWorkload<BackendT>::on_update();
      m_input->allocate(this->get_work_size());
    }

  private:
    std::shared_ptr<typename Kernel::Input> m_input;
    std::unique_ptr<Kernel> m_kernel;
    std::shared_ptr<typename Kernel::Output> m_gpu_output;
    std::shared_ptr<typename Kernel::Output> m_comparison_output;
  };
} // namespace Baseliner

#endif // KERNEL_HPP