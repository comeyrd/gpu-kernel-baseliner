#ifndef RUNNER_HPP
#define RUNNER_HPP
#include <baseliner/Executable.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/Stats.hpp>
#include <baseliner/StoppingCriterion.hpp>
#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <iostream>
#include <memory>
#include <type_traits>
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

  protected:
    void register_options() override {
      add_option("Runner", "block", "Using a blocking kernel", m_block);
      add_option("Runner", "block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("Runner", "flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("Runner", "warmup", "Having a warmup run", m_warmup);
    }

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
    // Runner
    explicit Runner()
        : IRunner(),
          m_input(std::make_shared<typename Kernel::Input>()),
          m_backend(),
          m_stream(m_backend.create_stream()),
          m_flusher(),
          m_blocker(),
          m_kernel(std::make_unique<Kernel>(m_input)),
          m_stats_engine(std::make_shared<Stats::StatsEngine>()),
          m_stopping(std::make_unique<StoppingCriterion>(m_stats_engine)) {};

    template <typename TStopping, typename... Args>
    auto set_stopping_criterion(Args &&...args) -> Runner & {
      static_assert(std::is_base_of_v<StoppingCriterion, TStopping>,
                    "TStopping must inherit from Baseliner::StoppingCriterion");
      m_stopping = std::make_unique<TStopping>(m_stats_engine, std::forward<Args>(args)...);
      return *this;
    }
    template <typename TStat, typename... Args>
    auto add_stat(Args &&...args) -> Runner & {
      static_assert(std::is_base_of_v<Stats::IStatBase, TStat>, "A Stat must inherit from Baseliner::Stats::IStatBase");
      m_stats_engine->register_stat<TStat>((args, ...));
      return *this;
    }

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
        m_stats_engine->update_values<Stats::ExecutionTime>(m_kernel->time_elapsed());
        m_stats_engine->compute_stats();
      }
      postAll();
      m_kernel->teardown(m_out_gpu);
      if (!(m_out_gpu == m_out_cpu)) {
        std::cout << "Error, GPU and CPU results are not the same" << '\n';
        // std::cout << m_out_gpu << std::endl << " | " << m_out_cpu << std::endl;
      }

      Result result(this->gather_options(), m_kernel->name());
      std::vector<Metric> metrics = {m_stats_engine->get_metrics()};
      result.push_back_metrics(metrics);
      return result;
    }

  protected:
    void register_dependencies() override {
      register_consumer(*m_input);
      register_consumer(*m_stopping);
    };

  private:
    // Kernel Types
    std::shared_ptr<Stats::StatsEngine> m_stats_engine;
    std::unique_ptr<StoppingCriterion> m_stopping;
    std::shared_ptr<typename Kernel::Input> m_input;

    // Stats registry
    // Backend specifics
    Device m_backend;
    typename Device::L2Flusher m_flusher;
    typename Device::BlockingKernel m_blocker;
    std::shared_ptr<typename Device::stream_t> m_stream;

    // Kernel Holder
    std::unique_ptr<Kernel> m_kernel;

    virtual void preAll() {
      m_stats_engine->register_stat<Stats::Repetitions>();
      m_stats_engine->register_stat<Stats::ExecutionTimeVector>();
      m_stats_engine->register_stat<Stats::SortedExecutionTimeVector>();
      m_stats_engine->register_stat<Stats::Q1>();
      m_stats_engine->register_stat<Stats::Q3>();
      m_stats_engine->register_stat<Stats::Median>();
      m_stats_engine->register_stat<Stats::MedianConfidenceInterval>();
      m_stats_engine->build_execution_plan();
      if (get_warmup()) {
        m_kernel->run(m_stream);
      }
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