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

  class IInput : public MoveOnly, public OptionConsumer {
  public:
    virtual void generate_random() = 0;
    virtual void resize(int work_size) = 0;

  protected:
    int m_work_size;
    IInput(int work_size)
        : m_work_size(work_size) {};
  };
  class IOutput : public MoveOnly {
  public:
    virtual void resize(const int work_size) = 0;

  protected:
    IOutput(int work_size) {};
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

  protected:
    const Input &m_input;
  };
} // namespace Baseliner

#endif // KERNEL_HPP