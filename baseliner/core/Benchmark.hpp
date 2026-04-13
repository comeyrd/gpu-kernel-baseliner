#ifndef BASELINER_CORE_BENCHMARK_HPP
#define BASELINER_CORE_BENCHMARK_HPP
#include <baseliner/core/BenchmarkReport.hpp>
#include <baseliner/core/Error.hpp>
#include <baseliner/core/IPrinter.hpp>
#include <baseliner/core/Kernel.hpp>
#include <baseliner/core/Metric.hpp>
#include <baseliner/core/OptionTypes.hpp>
#include <baseliner/core/Options.hpp>
#include <baseliner/core/State.hpp>
#include <baseliner/core/StoppingCriterion.hpp>
#include <baseliner/core/Workload.hpp>
#include <baseliner/core/hardware/Backend.hpp>
#include <baseliner/core/stats/IStats.hpp>
#include <baseliner/core/stats/Stats.hpp>
#include <baseliner/core/stats/StatsEngine.hpp>
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

    IBenchmark(IBenchmark &&) noexcept = default;
    auto operator=(IBenchmark &&) noexcept -> IBenchmark & = default;

    IBenchmark(const IBenchmark &) = delete;
    auto operator=(const IBenchmark &) -> IBenchmark & = delete;

    ~IBenchmark() override = default;
    virtual auto run_benchmark() -> BenchmarkReport = 0;

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

    virtual auto get_workload_options() -> OptionsMap = 0;

    void set_stopping_criterion(const std::function<std::unique_ptr<StoppingCriterion>()> &stopping_builder) {
      m_stopping = stopping_builder();
      m_stopping->set_stats_engine(m_stats_engine);
    }
    void add_stats(const std::vector<std::function<void(std::shared_ptr<Stats::StatsEngine>)>> &stats_recipes) {
      for (auto stat : stats_recipes) {
        add_stat(stat);
      }
    }
    void add_stat(std::function<void(std::shared_ptr<Stats::StatsEngine>)> &stat_recipe) {
      stat_recipe(m_stats_engine);
    }

    void set_stat_options(const OptionsMap &omap) {
      stats_options = omap;
    }
    [[nodiscard]] auto get_stat_options() const -> OptionsMap {
      return stats_options;
    }
    [[nodiscard]] auto get_sweep_spec() const -> std::optional<SweepSpec> {
      return m_sweep_spec;
    }
    void set_sweep_spec(const std::optional<SweepSpec> &spec) {
      m_sweep_spec = spec;
    }
    void set_backend_options(const OptionsMap &omap) {
      m_backend_options = omap;
    }
    void set_printer(std::shared_ptr<IBenchmarkPrinter> printer) {
      m_printer = printer;
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

    auto get_stopping_no_except() -> std::optional<StoppingCriterion *> {
      if (m_stopping) {
        return m_stopping.get();
      }
      return {};
    }
    auto get_stopping() -> StoppingCriterion * {
      return m_stopping.get();
    }
    auto get_stats_engine() -> Stats::StatsEngine * {
      return m_stats_engine.get();
    }
    auto get_stats_engine_shared() -> std::shared_ptr<Stats::StatsEngine> {
      return m_stats_engine;
    }

    auto get_backend_options() -> OptionsMap {
      return m_backend_options;
    }

    void print_callback(const SingleRunReport &report) {
      if (m_printer) {
        m_printer->consume_single_run_report(report);
      }
    }
    [[nodiscard]] virtual auto single_run(const std::optional<OptionsMap> &sweep_point) -> SingleRunReport = 0;

  private:
    bool m_warmup = true;
    bool m_flush_l2 = true;
    bool m_block = false;
    float m_block_duration_ms = DEFAULT_BLOCK_DURATION;
    bool m_time_setup = false;
    bool m_time_teardown = false;
    bool m_first = true;
    OptionsMap stats_options;
    std::string m_name{DEFAULT_BENCHMARK_NAME};
    std::unique_ptr<StoppingCriterion> m_stopping;
    std::shared_ptr<Stats::StatsEngine> m_stats_engine;
    std::optional<SweepSpec> m_sweep_spec;
    OptionsMap m_backend_options;
    std::shared_ptr<IBenchmarkPrinter> m_printer;
  };

  template <typename BackendT>
  class Benchmark : public IBenchmark {
  public:
    using backend = BackendT;
    // Benchmark
    explicit Benchmark()
        : IBenchmark() {
      get_stats_engine()->template register_stat<Stats::ExecutionTimeVector>();
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
      if (m_workload) {
        return m_workload->name() + get_m_name();
      }
      return get_m_name();
    }
    auto set_workload(std::shared_ptr<IWorkload<BackendT>> workload_impl) {
      m_workload = workload_impl;
    }
    auto run_benchmark() -> BenchmarkReport override {
      BenchmarkReport report;
      BackendT::instance()->apply_options(get_backend_options());
      report.hardware = this->get_hardware_info();
      auto sweeppoints = this->generate_sweep_points();
      for (const std::optional<OptionsMap> &sweep_point : sweeppoints) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        report.results.push_back(this->single_run(sweep_point));
        print_callback(report.results.back());
      }
      return report;
    }
    auto get_workload_options() -> OptionsMap override {
      return m_workload->get_options();
    };

  protected:
    auto generate_sweep_points() -> std::vector<OptionsMap> {
      if (!get_sweep_spec().has_value()) {
        return {{}};
      }
      SweepSpec spec = get_sweep_spec().value();
      auto resolved_axis = this->resolve_depedency_sweep_axis(spec.axes);
      return Sweep::get_sweep_points(spec.strategy, resolved_axis);
    }
    void register_options_dependencies() override {
      if (m_workload) {
        this->register_consumer(m_workload.get());
      }
      if (get_stopping_no_except().has_value()) {
        this->register_consumer(get_stopping());
      }
      this->register_consumer(BackendT::instance());
      this->register_consumer(get_stats_engine());
    }

    [[nodiscard]] auto single_run(const std::optional<OptionsMap> &sweep_point) -> SingleRunReport override {
      this->apply_sweep_point(sweep_point);
      m_stream = BackendT::instance()->create_stream();
      check_components();
      setup_metrics();
      get_stats_engine()->reset_engine();
      m_workload->setup(m_stream);
      update_metrics();
      pre_all();
      while (!get_stopping()->satisfied()) {
        if (ExecutionController::exit_requested()) {
          break;
        }
        m_workload->reset_workload(m_stream);
        pre_run();
        m_workload->timed_run(m_stream);
        post_run();
        get_stats_engine()->template update_values<Stats::ExecutionTime>(m_workload->time_elapsed());
        get_stats_engine()->compute_stats();
      }
      post_all();
      m_workload->teardown(m_stream);
      bool valid_run = m_workload->validate_workload();
      if (!valid_run) {
        std::cout << "Warning, not able to validate Case : " << m_workload->name() << '\n';
      }
      std::vector<Metric> metrics = {get_stats_engine()->get_metrics()};
      m_stream.reset();
      return SingleRunReport{sweep_point, metrics};
    }

  private:
    // Kernel Types

    // Stats registry
    // Hardware specifics
    Hardware::L2Flusher<BackendT> *m_flusher = Hardware::L2Flusher<BackendT>::instance();
    Hardware::BlockingKernel<BackendT> *m_blocker = Hardware::BlockingKernel<BackendT>::instance();
    std::shared_ptr<typename BackendT::stream_t> m_stream;

    std::shared_ptr<IWorkload<BackendT>> m_workload;

    [[nodiscard]] auto get_hardware_info() const -> Hardware::HardwareInfo {
      return backend::instance()->get_device_info();
    };
    void apply_sweep_point(const std::optional<OptionsMap> &point) {
      if (point.has_value()) {
        this->apply_depedencies_options(point.value());
      }
    }
    virtual void update_metrics() {
      if (m_workload) {
        m_workload->update_metrics(get_stats_engine_shared());
      }
    }
    virtual void setup_metrics() {
      if (get_first()) {
        if (get_warmup()) {
          get_stats_engine()->template register_metric<Stats::WarmupTime>();
        }
        if (get_timed_setup()) {
          get_stats_engine()->template register_metric<Stats::SetupTime>();
        }
        if (get_timed_teardown()) {
          get_stats_engine()->template register_metric<Stats::TeardownTime>();
        }
        if (m_workload) {
          m_workload->setup_metrics(get_stats_engine_shared());
        }
        get_stats_engine()->set_options(get_stat_options());
        set_first(false);
      }
    }
    virtual void pre_all() {
      if (get_warmup()) {
        m_workload->timed_run(m_stream);
        get_stats_engine()->template update_values<Stats::WarmupTime>(m_workload->time_elapsed());
      }
    };
    virtual void pre_run() {
      if (get_flush_l2()) {
        m_flusher->flush(m_stream);
      }
      if (get_block()) {
        m_blocker->block(m_stream, get_block_duration());
      }
    };
    virtual void post_run() {
      BackendT::get_last_error();
      if (get_block()) {
        m_blocker->unblock();
      }
    };
    virtual void post_all() {
      BackendT::synchronize(m_stream);
    };
    void check_components() {
      if (!m_workload) {
        throw Errors::empty_workload_benchmark();
      }
      if (!get_stopping_no_except().has_value()) {
        throw Errors::empty_stopping_benchmark();
      }
    }
  };
} // namespace Baseliner

#endif // RUNNER_HPP