#ifndef KERNEL_HPP
#define KERNEL_HPP
#include "Options.hpp"
#include <memory>
namespace Baseliner {

  // Move only semantics base class
  class MoveOnly {
  protected:
    MoveOnly() = default;

  public:
    MoveOnly(const MoveOnly &) = delete;
    MoveOnly &operator=(const MoveOnly &) = delete;
    MoveOnly(MoveOnly &&) noexcept = default;
    MoveOnly &operator=(MoveOnly &&) noexcept = default;
    virtual ~MoveOnly() = default;
  };
  // TODO Work on the instantiation of InputData : Reusing old data and saving data to file.
  class IInput : public MoveOnly, public OptionConsumer {
  public:
    void register_options() override {
      add_option("Kernel", "work_size", "The multiplier of the base work size to apply to the kernel", m_work_size);
      add_option("Kernel", "seed", "The seed used for the generation of input data", seed);
    }
    virtual void generate_random() = 0;

  protected:
    virtual void allocate() = 0;
    int m_work_size = 1;
    int seed = 202;
    IInput() = default;
    virtual ~IInput() = default;
  };
  template <typename Input>
  class IOutput : public MoveOnly {
  public:
  protected:
    const Input &m_input;
    IOutput(const Input &input)
        : m_input(input) {};
    virtual ~IOutput() = default;
  };

  template <typename stream_t, typename I, typename O>
  class IKernel {
  public:
    using Input = I;
    using Output = O;
    virtual void cpu(Output &output) = 0;
    virtual void setup() = 0;
    virtual void reset() = 0;
    virtual void run(std::shared_ptr<stream_t> &stream) = 0;
    virtual void teardown(Output &output) = 0;
    IKernel(const Input &input)
        : m_input(input) {};
    virtual ~IKernel() = default;

  protected:
    const Input &m_input;
  };
} // namespace Baseliner

#endif // KERNEL_HPP