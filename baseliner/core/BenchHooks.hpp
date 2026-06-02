#ifndef BASELINER_CORE_BENCHHOOKS_HPP
#define BASELINER_CORE_BENCHHOOKS_HPP
#include <baseliner/core/stats/StatsEngine.hpp>
#include <memory>
namespace Baseliner {
  class IBenchmark;

  class IBenchHook {
  public:
    virtual void pre_all() = 0;
    virtual void pre_trial() = 0;
    virtual void pre_batch() = 0;
    virtual void post_batch() = 0;
    virtual void post_trial() = 0;
    virtual void post_all() = 0;
    virtual ~IBenchHook() = default;

    void set_engine(std::shared_ptr<Stats::StatsEngine> &engine) {
      m_engine = engine;
    };

    void set_benchmark(IBenchmark *benchmark) {
      m_benchmark = benchmark;
    };

  protected:
    std::shared_ptr<Stats::StatsEngine> m_engine;
    IBenchmark *m_benchmark;
  };

} // namespace Baseliner

#endif // BASELINER_CORE_BENCHHOOKS_HPP