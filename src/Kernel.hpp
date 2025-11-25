#ifndef KERNEL_HPP
#define KERNEL_HPP
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

  template <typename stream_t>
  class IKernel {
  public:
    class Input : public MoveOnly {
    public:
      virtual void generate_random() = 0;
      virtual void resize(const int work_size) = 0;

    protected:
      explicit Input(const int work_size) {};
    };
    class Output : public MoveOnly {
    public:
      virtual void resize(const int work_size) = 0;

    protected:
      explicit Output(const int work_size) {};
    };

    template<typename Input_,typename Output_>
    class GpuImplementation {
    public:
      virtual void setup() = 0;
      virtual void reset() = 0;
      virtual void run(stream_t &stream) = 0;
      virtual void teardown(Output_ &output) = 0;
      GpuImplementation(const Input_ &input) : m_input(input) {};

    protected:
      const Input_ &m_input;
    };
  };

} // namespace Baseliner

#endif // KERNEL_HPP