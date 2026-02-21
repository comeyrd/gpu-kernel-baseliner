#ifndef KERNEL_HPP
#define KERNEL_HPP
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
  constexpr int DEFAULT_SEED = 202;
  // TODO Work on the instantiation of InputData : Reusing old data and saving data to file.
  class IInput : public MoveOnly, public IOption {
  public:
    virtual void generate_random() = 0;

    ~IInput() override = default;
    IInput() = default;

    auto get_work_size() const -> int {
      return m_work_size;
    }
    auto get_seed() const -> int {
      return seed;
    }

  protected:
    void register_options() override {
      add_option("Kernel", "work_size", "The multiplier of the base work size to apply to the kernel", m_work_size);
      add_option("Kernel", "seed", "The seed used for the generation of input data", seed);
    }

  private:
    virtual void allocate() = 0;
    int m_work_size = 1;
    int seed = DEFAULT_SEED;
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
    using Kernel_Backend = B;
    virtual void setup(std::shared_ptr<typename Kernel_Backend::stream_t> stream) = 0;
    virtual void reset_kernel(std::shared_ptr<typename Kernel_Backend::stream_t> stream) = 0;
    virtual void setup_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void update_metrics(std::shared_ptr<Stats::StatsEngine> &engine) {};
    virtual void run(std::shared_ptr<typename Kernel_Backend::stream_t> stream) = 0;
    virtual void teardown(std::shared_ptr<typename Kernel_Backend::stream_t> stream, Output &output) = 0;
    virtual auto name() -> std::string = 0;
    virtual auto number_of_floating_point_operations() -> std::optional<size_t> {
      return {};
    }
    virtual auto number_of_bytes() -> std::optional<size_t> {
      return {};
    }
    IKernel(const std::shared_ptr<const Input> input)
        : m_input(input) {};
    virtual ~IKernel() = default;

    auto get_input() -> std::shared_ptr<const Input> {
      return m_input;
    };

  private:
    std::shared_ptr<const Input> m_input;
  };
} // namespace Baseliner

#endif // KERNEL_HPP