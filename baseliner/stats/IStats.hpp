#ifndef BASELINER_ISTATS_HPP
#define BASELINER_ISTATS_HPP

#include <baseliner/Metric.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <string>
#include <typeindex>
#include <vector>

namespace Baseliner::Stats {

  class IStatBase {
  public:
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    virtual ~IStatBase() = default;

    // What types do i need
    [[nodiscard]] virtual auto dependencies() const -> std::vector<std::type_index> = 0;

    // What types do i provide
    [[nodiscard]] virtual auto output() const -> std::type_index = 0;

    virtual void compute(StatsRegistry &reg) = 0;
    [[nodiscard]] virtual auto get_value(const StatsRegistry &reg) const -> MetricData = 0;

    IStatBase() = default;

  private:
  };

  // Helper for Tag management.
  template <typename OutputTag, typename ValueType, typename... InputTags>
  class IStat : public IStatBase {
  public:
    using type = ValueType;
    using InputTuple = std::tuple<InputTags...>;
    [[nodiscard]] auto dependencies() const -> std::vector<std::type_index> override {
      return {std::type_index(typeid(InputTags))...};
    }

    auto get_value(const StatsRegistry &reg) const -> MetricData override {
      if (!reg.has<OutputTag>()) {
        return {}; // Returns the first type in variant (monostate/default)
      }

      // Because ValueType is one of the types in your MetricData variant,
      // this assignment works automatically.
      return reg.get<OutputTag>();
    }

    [[nodiscard]] auto output() const -> std::type_index override {
      return std::type_index(typeid(OutputTag));
    }
    virtual void calculate(ValueType &value_to_update, const typename InputTags::type &...inputs) = 0;

    void compute(StatsRegistry &reg) final {
      if (!reg.has<OutputTag>()) {
        reg.set<OutputTag>(ValueType{});
      }
      if (!(reg.has<InputTags>() && ...)) {
      }
      calculate(reg.get_mutable<OutputTag>(), reg.get<InputTags>()...);
    };
  };

} // namespace Baseliner::Stats

#endif // __BASELINER__ISTATS_HPP
