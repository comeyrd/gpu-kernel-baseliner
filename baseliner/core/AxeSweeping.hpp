#ifndef BASELINER_CORE_AXESWEEPING_HPP
#define BASELINER_CORE_AXESWEEPING_HPP
#include <baseliner/core/OptionTypes.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Baseliner {
  enum class SweepStrategy : char {
    FullGrid
  };
  DESCRIBE_ENUM(SweepStrategy, ENUM_VALUE(FullGrid))

  enum class SweepPolicy : char {
    PowersOfTwo,
    LinearRange,
    Enumerated
  };
  DESCRIBE_ENUM(SweepPolicy, ENUM_VALUE(PowersOfTwo), ENUM_VALUE(LinearRange), ENUM_VALUE(Enumerated))

  struct SweepHint {
    SweepPolicy policy;
    std::string min;
    std::string max;
    std::string step;
    std::vector<std::string> enumerated;
  };
  DESCRIBE(SweepHint, FIELD(policy), FIELD(min), FIELD(max), FIELD(step), FIELD(enumerated))

  template <typename T>
  struct TypedSweepHint {
    SweepPolicy policy;
    T min;
    T max;
    T step;
    std::vector<T> enumerated;
  };

  struct SweepAxis {
    std::string interface;
    std::string option;
    std::optional<SweepHint> hint;
  };
  DESCRIBE(SweepAxis, FIELD(interface), FIELD(option), FIELD(hint))

  struct ResolvedAxis {
    std::string interface;
    std::string option;
    std::vector<std::string> value;
  };
  DESCRIBE(ResolvedAxis, FIELD(interface), FIELD(option), FIELD(value))

  struct SweepSpec {
    SweepStrategy strategy;
    std::vector<SweepAxis> axes;
  };
  DESCRIBE(SweepSpec, FIELD(strategy), FIELD(axes))

  using SweepHintMap = std::unordered_map<std::string, std::unordered_map<std::string, SweepHint>>;

  namespace Sweep {

    namespace Detail {

      template <typename T>
      struct Sweeper {
        static auto generate(const TypedSweepHint<T> &hint) -> std::vector<T> {
          if (hint.policy == SweepPolicy::Enumerated) {
            return hint.enumerated;
          }

          std::vector<T> result;
          if (hint.policy == SweepPolicy::LinearRange) {
            if (hint.step <= static_cast<T>(0)) {
              throw std::invalid_argument("Step must be > 0");
            }
            for (T value = hint.min; value <= hint.max; value += hint.step) {
              result.push_back(value);
            }
          } else if (hint.policy == SweepPolicy::PowersOfTwo) {
            if (hint.min <= static_cast<T>(0)) {
              throw std::invalid_argument("Min must be > 0");
            }
            for (T value = hint.min; value <= hint.max; value *= static_cast<T>(2)) {
              result.push_back(value);
            }
          }
          return result;
        }
      };

      template <>
      struct Sweeper<bool> {
        static auto generate(const TypedSweepHint<bool> &hint) -> std::vector<bool> {
          if (hint.policy == SweepPolicy::Enumerated) {
            return hint.enumerated;
          }
          if (hint.min == hint.max) {
            return {hint.min};
          }
          return {false, true};
        }
      };
    } // namespace Detail

    template <typename T>
    auto generate_sweep_values(const TypedSweepHint<T> &typed) -> std::vector<T> {
      return Detail::Sweeper<T>::generate(typed);
    }

    inline auto get_sweep_points(const SweepStrategy &strategy, const std::vector<ResolvedAxis> &axes)
        -> std::vector<OptionsMap> {

      if (axes.empty()) {
        return {};
      }

      std::vector<OptionsMap> result = {{}}; // Start with one empty map

      switch (strategy) {
      case SweepStrategy::FullGrid: {
        for (const ResolvedAxis &axis : axes) {
          std::vector<OptionsMap> next;

          for (const OptionsMap &existing : result) {
            for (const std::string &val : axis.value) {
              OptionsMap entry = existing; // copy current combination
              entry[axis.interface][axis.option] = Option{"", val};
              next.push_back(std::move(entry));
            }
          }

          result = std::move(next);
        }
        break;
      }
      }

      return result;
    }
  } // namespace Sweep
} // namespace Baseliner

#endif