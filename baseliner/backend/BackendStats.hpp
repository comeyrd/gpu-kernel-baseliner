#ifndef BASELINER_BACKEND_STATS_HPP
#define BASELINER_BACKEND_STATS_HPP
#include <baseliner/stats/Stats.hpp>

namespace Baseliner::Stats {

  template <typename BackendT>
  class ClockFrequency : public IStat<ClockFrequency<BackendT>, float> {
  public:
    [[nodiscard]] auto name() const -> std::string override;
    [[nodiscard]] auto unit() const -> std::string override {
      return "GHz";
    }
    void calculate(typename ClockFrequency<BackendT>::type &value_to_update) override;
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };

  template <typename BackendT>
  class ClockFrequencyVector
      : public IStat<ClockFrequencyVector<BackendT>, std::vector<float>, ClockFrequency<BackendT>> {
  public:
    [[nodiscard]] auto name() const -> std::string override;
    [[nodiscard]] auto unit() const -> std::string override {
      return "GHz";
    }
    void calculate(typename ClockFrequencyVector<BackendT>::type &value_to_update,
                   typename ClockFrequency<BackendT>::type const &input) override {
      value_to_update.push_back(input);
    };
    [[nodiscard]] auto compute_policy() -> StatComputePolicy override {
      return StatComputePolicy::EVERY_TICK;
    };
  };
} // namespace Baseliner::Stats
#endif // BASELINER_BACKEND_STATS_HPP