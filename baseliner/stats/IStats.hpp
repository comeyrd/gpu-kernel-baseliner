#ifndef BASELINER_ISTATS_HPP
#define BASELINER_ISTATS_HPP

#include <baseliner/Metric.hpp>
#include <baseliner/Options.hpp>
#include <baseliner/stats/StatsRegistry.hpp>
#include <string>
#include <typeindex>
#include <vector>

namespace Baseliner::Stats {
  enum StatComputePolicy : u_int8_t {
    EVERY_TICK,
    ON_DEMAND
  };

  class IStatBase : public LazyOption {
  public:
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    virtual ~IStatBase() = default;

    // What types do i need
    [[nodiscard]] virtual auto dependencies() const -> std::vector<std::type_index> = 0;

    // What types do i provide
    [[nodiscard]] virtual auto output() const -> std::type_index = 0;

    // When do i need to refresh
    [[nodiscard]] virtual auto compute_policy() -> StatComputePolicy = 0;

    virtual void compute(StatsRegistry &reg) = 0;
    [[nodiscard]] virtual auto get_value(const StatsRegistry &reg) const -> MetricData = 0;
    virtual void set_default(StatsRegistry &reg) = 0;
    IStatBase() = default;

  private:
  };

  // Helper for Tag management.
  template <typename OutputTag, typename ValueType, typename... InputTags>
  class IStat : public IStatBase {
  public:
    using type = ValueType;
    using tuple = std::tuple<InputTags...>;
    [[nodiscard]] auto dependencies() const -> std::vector<std::type_index> override {
      return {std::type_index(typeid(InputTags))...};
    }

    [[nodiscard]] auto get_value(const StatsRegistry &reg) const -> MetricData override {
      if (!reg.has<OutputTag>()) {
        return {};
      }
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
        std::stringstream sstream;
        sstream << "Error in stat compute graph, one of inputs : ";
        bool first = true;
        ((sstream << (first ? "" : ", ") << typeid(InputTags).name(), first = false), ...);
        sstream << "are not yet computed by needed by " << typeid(OutputTag).name();
        throw std::runtime_error(sstream.str());
      }
      calculate(reg.get_mutable<OutputTag>(), reg.get<InputTags>()...);
    };
    void set_default(StatsRegistry &reg) override {
      reg.set<OutputTag>(m_default_value);
    }
    IStat()
        : IStatBase(),
          m_default_value{} {};

    IStat(ValueType default_value)
        : IStatBase(),
          m_default_value(default_value) {};

  private:
    ValueType m_default_value;
  };

  class IMetricBase : public LazyOption {
  public:
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    [[nodiscard]] virtual auto unit() const -> std::string = 0;
    [[nodiscard]] virtual auto output() const -> std::type_index = 0;
    [[nodiscard]] virtual auto get_value(const StatsRegistry &reg) const -> MetricData = 0;
    virtual ~IMetricBase() = default;
  };
  template <typename OutputTag, typename ValueType>
  class Imetric : public IMetricBase {
  public:
    using type = ValueType;
    [[nodiscard]] auto get_value(const StatsRegistry &reg) const -> MetricData override {
      if (!reg.has<OutputTag>()) {
        return {}; // Returns the first type in variant (monostate/default)
      }
      return reg.get<OutputTag>();
    }
    [[nodiscard]] auto output() const -> std::type_index override {
      return std::type_index(typeid(OutputTag));
    }
  };

} // namespace Baseliner::Stats

#endif // __BASELINER__ISTATS_HPP
