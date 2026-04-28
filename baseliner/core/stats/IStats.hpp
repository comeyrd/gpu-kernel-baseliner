#ifndef BASELINER_CORE_STATS_ISTATS_HPP
#define BASELINER_CORE_STATS_ISTATS_HPP

#include <baseliner/core/Metric.hpp>
#include <baseliner/core/Options.hpp>

#include <baseliner/core/stats/StatsRegistry.hpp>
#include <string>
#include <typeindex>
#include <vector>

namespace Baseliner::Stats {
  enum SavingPolicy : char {
    SAVE,
    DISCARD
  };

  class IStatBase : public LazyOption {
  public:
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    [[nodiscard]] virtual auto unit() const -> std::string = 0;
    virtual ~IStatBase() = default;

    // What types do i need
    [[nodiscard]] virtual auto dependencies() const -> std::vector<std::type_index> = 0;

    // What types do i provide
    [[nodiscard]] virtual auto output() const -> std::type_index = 0;

    // When do i need to refresh
    [[nodiscard]] virtual auto granularity() const -> MetricGranularity = 0;

    // Does the Stats needs to be saved
    [[nodiscard]] virtual auto saving_policy() const -> SavingPolicy {
      return SavingPolicy::SAVE;
    };

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
      if (this->saving_policy() == SavingPolicy::SAVE) {
        return reg.get<OutputTag>();
      }
      return std::monostate();
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
        bool first = true;
        ((sstream << (first ? "" : ", ") << typeid(InputTags).name(), first = false), ...);

        throw Errors::stat_dependencies_not_yet_computed(sstream.str(), typeid(OutputTag).name());
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
    virtual void set_default(StatsRegistry &reg) = 0;
    [[nodiscard]] virtual auto saving_policy() const -> SavingPolicy {
      return SavingPolicy::SAVE;
    };
    [[nodiscard]] virtual auto granularity() const -> MetricGranularity = 0;
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
    void set_default(StatsRegistry &reg) override {
      reg.set<OutputTag>(m_default_value);
    }
    Imetric()
        : IMetricBase(),
          m_default_value{} {};

    Imetric(ValueType default_value)
        : IMetricBase(),
          m_default_value(default_value) {};

  private:
    ValueType m_default_value;
  };

} // namespace Baseliner::Stats

#endif // __BASELINER__ISTATS_HPP
