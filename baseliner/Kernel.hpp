#ifndef KERNEL_HPP
#define KERNEL_HPP
#include "baseliner/Case.hpp"
#include "baseliner/stats/StatsEngine.hpp"
#include <baseliner/Options.hpp>
#include <baseliner/Timer.hpp>
#include <baseliner/backend/Backend.hpp>
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
  class IInput : public MoveOnly, public IBaseCase {
  public:
    virtual void generate_random() = 0;

    ~IInput() override = default;
    IInput() = default;

  private:
    virtual void allocate() = 0;
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
    virtual void setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void run(std::shared_ptr<typename backend::stream_t> stream) = 0;
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
    void case_setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      m_kernel->setup_metrics(engine);
    };
    void case_update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) override {
      m_kernel->update_metrics(engine);
    };
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

    auto number_of_floating_point_operations() -> std::optional<size_t> override {
      return m_input->number_of_floating_point_operations();
    }
    auto number_of_bytes() -> std::optional<size_t> override {
      return m_input->number_of_bytes();
    }

  private:
    std::shared_ptr<typename Kernel::Input> m_input;
    std::unique_ptr<Kernel> m_kernel;
    std::shared_ptr<typename Kernel::Output> m_gpu_output;
    std::shared_ptr<typename Kernel::Output> m_comparison_output;
  };
} // namespace Baseliner

#endif // KERNEL_HPP