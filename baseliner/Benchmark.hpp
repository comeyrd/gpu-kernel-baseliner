#ifndef BASELINER_BENCHMARK_HPP
#define BASELINER_BENCHMARK_HPP
#include "baseliner/backend/Backend.hpp"
#include <baseliner/Case.hpp>
#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/Result.hpp>
#include <baseliner/StoppingCriterion.hpp>

#include <baseliner/stats/IStats.hpp>
#include <baseliner/stats/Stats.hpp>
#include <baseliner/stats/StatsDictionnary.hpp>
#include <baseliner/stats/StatsEngine.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
inline static const std::string_view DEFAULT_BENCHMARK_NAME = "Benchmark";

#define BASELINER_BENCHMARK_SETTER(name, type)                                                                         \
  auto set_##name(type value) &->Benchmark & {                                                                         \
    this->set_m_##name(value); /* or call an internal logic function */                                                \
    return *this;                                                                                                      \
  }                                                                                                                    \
                                                                                                                       \
  /* The R-value version: calls the one above and moves *this */                                                       \
  auto set_##name(type value) &&->Benchmark {                                                                          \
    this->set_m_##name(value); /* Calls the & version */                                                               \
    return std::move(*this);                                                                                           \
  }

namespace Baseliner {
  constexpr float DEFAULT_BLOCK_DURATION = 1000.0F;
  class IBenchmark : public IOption {
  public:
    // Benchmark Options
    // IOption Interface
    explicit IBenchmark()
        : m_stats_engine(std::make_shared<Stats::StatsEngine>()) {};
    virtual auto name() -> std::string = 0;
    virtual auto run() -> Result = 0;

    IBenchmark(IBenchmark &&) noexcept = default;
    auto operator=(IBenchmark &&) noexcept -> IBenchmark & = default;

    IBenchmark(const IBenchmark &) = delete;
    auto operator=(const IBenchmark &) -> IBenchmark & = delete;

    ~IBenchmark() override = default;

    void set_m_warmup(bool warmup) {
      m_warmup = warmup;
    };
    [[nodiscard]] auto get_warmup() const -> bool {
      return m_warmup;
    };
    void set_m_flush_l2(bool flush_l2) {
      m_flush_l2 = flush_l2;
    };
    [[nodiscard]] auto get_flush_l2() const -> bool {
      return m_flush_l2;
    };
    void set_m_block(bool block) {
      m_block = block;
    };
    [[nodiscard]] auto get_block() const -> bool {
      return m_block;
    };
    void set_m_block_duration(float block_duration) {
      m_block_duration_ms = block_duration;
    };
    [[nodiscard]] auto get_block_duration() const -> float {
      return m_block_duration_ms;
    };
    void set_m_timed_setup(bool timed_setup) {
      m_time_setup = timed_setup;
    };
    [[nodiscard]] auto get_timed_setup() const -> bool {
      return m_time_setup;
    };
    void set_m_timed_teardown(bool timed_teardown) {
      m_time_teardown = timed_teardown;
    };
    [[nodiscard]] auto get_timed_teardown() const -> bool {
      return m_time_teardown;
    };
    [[nodiscard]] auto get_first() const -> bool {
      return m_first;
    }
    void set_m_first(bool first) {
      m_first = first;
    }
    [[nodiscard]] auto get_m_name() const -> std::string {
      return m_name;
    }
    void set_m_name(std::string name) {
      m_name = std::move(name);
    }

    void set_stopping_criterion(
        std::function<std::unique_ptr<StoppingCriterion>(std::shared_ptr<Stats::StatsEngine>)> stopping_builder) {
      m_stopping = stopping_builder(m_stats_engine);
    }

  protected:
    void register_options() override {
      add_option("Benchmark", "block", "Using a blocking kernel", m_block);
      add_option("Benchmark", "block_duration", "Duration of the blocking kernel (in ms)", m_block_duration_ms);
      add_option("Benchmark", "flush", "Enables the flushing of the L2 cache", m_flush_l2);
      add_option("Benchmark", "warmup", "Having a warmup run", m_warmup);
      add_option("Benchmark", "timed_setup", "Time the setup", m_time_setup);
      add_option("Benchmark", "timed_teardown", "Time the teardown", m_time_teardown);
    }
    std::unique_ptr<StoppingCriterion> m_stopping;
    std::shared_ptr<Stats::StatsEngine> m_stats_engine;

  private:
    bool m_warmup = true;
    bool m_flush_l2 = true;
    bool m_block = false;
    float m_block_duration_ms = DEFAULT_BLOCK_DURATION;
    bool m_time_setup = false;
    bool m_time_teardown = false;
    bool m_first = true;
    std::string m_name{DEFAULT_BENCHMARK_NAME};
  };

  template <typename BackendT>
  class Benchmark : public IBenchmark {
  public:
    using backend = BackendT;
    // Benchmark
    explicit Benchmark()
        : IBenchmark(),
          m_backend(),
          m_stream(m_backend.create_stream()),
          m_flusher(),
          m_blocker() {
      m_stats_engine->register_stat<Stats::ExecutionTimeVector>();
    };
    Benchmark(Benchmark &&) noexcept = default;
    auto operator=(Benchmark &&) noexcept -> Benchmark & = default;

    // Ensure copies are still deleted (good for safety)
    Benchmark(const Benchmark &) = delete;
    auto operator=(const Benchmark &) -> Benchmark & = delete;

    BASELINER_BENCHMARK_SETTER(warmup, bool);
    BASELINER_BENCHMARK_SETTER(name, std::string);
    BASELINER_BENCHMARK_SETTER(block, bool);
    BASELINER_BENCHMARK_SETTER(block_duration, float);
    BASELINER_BENCHMARK_SETTER(flush_l2, bool);
    BASELINER_BENCHMARK_SETTER(timed_setup, float);
    BASELINER_BENCHMARK_SETTER(timed_teardown, float);
    BASELINER_BENCHMARK_SETTER(first, bool);

    auto name() -> std::string override {
      if (m_case) {
        return m_case->name() + get_m_name();
      }
      return get_m_name();
    }
    template <typename TStopping, typename... Args>
    auto set_stopping_criterion(Args &&...args) & -> Benchmark & {
      static_assert(std::is_base_of_v<StoppingCriterion, TStopping>,
                    "TStopping must inherit from Baseliner::StoppingCriterion");
      m_stopping = std::make_unique<TStopping>(m_stats_engine, std::forward<Args>(args)...);
      return *this;
    }
    template <typename TStopping, typename... Args>
    auto set_stopping_criterion(Args &&...args) && -> Benchmark {
      this->set_stopping_criterion<TStopping>(std::forward<Args>(args)...);
      return std::move(*this);
    }

    template <typename TKernel, typename... Args>
    auto set_kernel(Args &&...args) & -> Benchmark & {
      m_case = std::make_shared<KernelCase<TKernel>>(std::forward<Args>(args)...);
      return *this;
    }

    template <typename TKernel, typename... Args>
    auto set_kernel(Args &&...args) && -> Benchmark {
      this->set_kernel<TKernel>(std::forward<Args>(args)...);
      return std::move(*this);
    }
    template <typename TCase, typename... Args>
    auto set_case(Args &&...args) & -> Benchmark & {
      static_assert(std::is_base_of_v<ICase<BackendT>, TCase>,
                    "TStopping must inherit from Baseliner::StoppingCriterion");
      m_case = std::make_shared<TCase>(std::forward<Args>(args)...);
      return *this;
    }

    template <typename TCase, typename... Args>
    auto set_case(Args &&...args) && -> Benchmark {
      this->set_case<TCase>(std::forward<Args>(args)...);
      return std::move(*this);
    }
    auto set_case(std::shared_ptr<ICase<BackendT>> case_impl) & -> Benchmark & {
      m_case = case_impl;
      return *this;
    }

    auto set_case(std::shared_ptr<ICase<BackendT>> case_impl) && -> Benchmark {
      this->set_case(case_impl);
      return std::move(*this);
    }

    template <typename TStat, typename... Args>
    auto add_stat(Args &&...args) & -> Benchmark & {
      static_assert(std::is_base_of_v<Stats::IStatBase, TStat>, "A Stat must inherit from Baseliner::Stats::IStatBase");
      m_stats_engine->register_stat<TStat>(std::forward<Args>(args)...);
      return *this;
    }
    template <typename TStat, typename... Args>
    auto add_stat(Args &&...args) && -> Benchmark {
      this->add_stat<TStat>(std::forward<Args>(args)...);
      return std::move(*this);
    }
    void add_stats(std::vector<std::string> &stats_names) {
      for (auto stat : stats_names) {
        add_stat(stat);
      }
    }
    void add_stat(std::string &statname) {
      Stats::StatsDictionnary::instance()->add_stat_to_engine(statname, m_stats_engine);
    }

    auto run() -> Result override {
      check_case();
      setup_metrics();
      m_stats_engine->reset_engine();
      m_case->setup(m_stream);
      update_metrics();
      pre_all();
      while (!m_stopping->satisfied()) {
        m_case->reset_case(m_stream);
        pre_run();
        m_case->timed_run(m_stream);
        post_run();
        m_stats_engine->update_values<Stats::ExecutionTime>(m_case->time_elapsed());
        m_stats_engine->compute_stats();
      }
      post_all();
      m_case->teardown(m_stream);
      bool valid_run = m_case->validate_case();
      if (!valid_run) {
        std::cout << "Warning, not able to validate Case : " << m_case->name() << '\n';
      }

      Result result(this->gather_options(), m_case->name(), valid_run);
      std::vector<Metric> metrics = {m_stats_engine->get_metrics()};
      result.push_back_metrics(metrics);
      return result;
    }

  protected:
    void register_options_dependencies() override {
      register_consumer(*m_stopping);
      register_consumer(*m_stats_engine);
      if (m_case) {
        register_consumer(*m_case);
      }
    };

  private:
    // Kernel Types

    // Stats registry
    // Backend specifics
    BackendT m_backend;
    Backend::L2Flusher<BackendT> m_flusher;
    Backend::BlockingKernel<BackendT> m_blocker;
    std::shared_ptr<typename BackendT::stream_t> m_stream;

    std::shared_ptr<ICase<BackendT>> m_case;
    virtual void update_metrics() {
      if (m_case) {
        m_case->update_metrics(m_stats_engine);
      }
    }
    virtual void setup_metrics() {
      if (get_first()) {
        if (get_warmup()) {
          m_stats_engine->register_metric<Stats::WarmupTime>();
        }
        if (get_timed_setup()) {
          m_stats_engine->register_metric<Stats::SetupTime>();
        }
        if (get_timed_teardown()) {
          m_stats_engine->register_metric<Stats::TeardownTime>();
        }
        if (m_case) {
          m_case->setup_metrics(m_stats_engine);
        }
        set_first(false);
      }
    }
    virtual void pre_all() {
      if (get_warmup()) {
        m_case->timed_run(m_stream);
        m_stats_engine->update_values<Stats::WarmupTime>(m_case->time_elapsed());
      }
    };
    virtual void pre_run() {
      if (get_flush_l2()) {
        m_flusher.flush(m_stream);
      }
      if (get_block()) {
        m_blocker.block(m_stream, get_block_duration());
      }
    };
    virtual void post_run() {
      m_backend.get_last_error();
      if (get_block()) {
        m_blocker.unblock();
      }
    };
    virtual void post_all() {
      m_backend.synchronize(m_stream);
    };
    void check_case() {
      if (!m_case) {
        throw std::runtime_error("Benchmark Error : Launching benchmark with an empty case");
      }
    }
  };
} // namespace Baseliner

#endif // RUNNER_HPP