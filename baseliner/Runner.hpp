#ifndef RUNNER_HPP
#define RUNNER_HPP
#include <baseliner/Executable.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
namespace Baseliner {
  constexpr float DEFAULT_BLOCK_DURATION = 1000.0F;
  class IRunner : public IOptionConsumer, public ISingleExecutable {
  public:
    // Runner Options
    // IOptionConsumer Interfac
    explicit IRunner() = default;

    ~IRunner() override = default;

    void register_options() override {
      add_option("Runner", "block", "Using a blocking kernel", m_block);
      add_option("Runner", "block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("Runner", "flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("Runner", "warmup", "Having a warmup run", m_warmup);
    }
    void set_warmup(bool warmup) {
      m_warmup = warmup;
    };
    auto get_warmup() const -> bool {
      return m_warmup;
    };
    void set_flush_l2(bool flush_l2) {
      m_flush_l2 = flush_l2;
    };
    auto get_flush_l2() const -> bool {
      return m_flush_l2;
    };
    void set_block(bool block) {
      m_block = block;
    };
    auto get_block() const -> bool {
      return m_block;
    };
    void set_block_duration(float block_duration) {
      m_block_duration_ms = block_duration;
    };
    auto get_block_duration() const -> float {
      return m_block_duration_ms;
    };

  private:
    bool m_warmup = true;
    bool m_flush_l2 = true;
    bool m_block = false;
    float m_block_duration_ms = DEFAULT_BLOCK_DURATION;
  };
  template <typename Kernel, typename Device>
  // TODO Setup the static checks at compile time.
  class Runner : public IRunner {
  public:
    void register_dependencies() override {
      register_consumer(*m_input);
      register_consumer(*m_stopping);
    };
    // Runner
    explicit Runner(std::unique_ptr<StoppingCriterion> stopping)
        : IRunner(),
          m_stopping(std::move(stopping)),
          m_input(std::make_shared<typename Kernel::Input>()),
          m_backend(),
          m_stream(m_backend.create_stream()),
          m_flusher(),
          m_blocker(),
          m_kernel(std::make_unique<Kernel>(m_input)) {};

    auto run() -> Result override {
      m_input->generate_random();
      typename Kernel::Output m_out_cpu(m_input);
      typename Kernel::Output m_out_gpu(m_input);
      m_kernel->cpu(m_out_cpu);
      m_kernel->setup();
      preAll();
      while (!m_stopping->satisfied()) {
        m_kernel->reset();
        preRun();
        m_kernel->timed_run(m_stream);
        postRun();
        m_stopping->addTime(m_kernel->time_elapsed());
      }
      postAll();
      m_kernel->teardown(m_out_gpu);
      if (!(m_out_gpu == m_out_cpu)) {
        std::cout << "Error, GPU and CPU results are not the same" << '\n';
        // std::cout << m_out_gpu << std::endl << " | " << m_out_cpu << std::endl;
      }

      Result result(this->gather_options(), m_kernel->name());
      std::vector<Metric> metrics = m_stopping->get_metrics();
      result.push_back_metrics(metrics);
      return result;
    }

  private:
    // Kernel Types
    std::unique_ptr<StoppingCriterion> m_stopping;
    std::shared_ptr<typename Kernel::Input> m_input;

    // Backend specifics
    Device m_backend;
    typename Device::L2Flusher m_flusher;
    typename Device::BlockingKernel m_blocker;
    std::shared_ptr<typename Device::stream_t> m_stream;

    // Kernel Holder
    std::unique_ptr<Kernel> m_kernel;

    virtual void preAll() {
      if (get_warmup()) {
        m_kernel->run(m_stream);
      }
      m_stopping->reset();
    };
    virtual void preRun() {
      if (get_flush_l2()) {
        m_flusher.flush(m_stream);
      }
      if (get_block()) {
        m_blocker.block(m_stream, get_block_duration());
      }
    };
    virtual void postRun() {
      m_backend.get_last_error();
      if (get_block()) {
        m_blocker.unblock();
      }
    };
    virtual void postAll() {
      m_backend.synchronize(m_stream);
    };
  };
} // namespace Baseliner

#endif // RUNNER_HPP