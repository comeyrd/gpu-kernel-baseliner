#ifndef __BASELINER__ISTATS_HPP
#define __BASELINER__ISTATS_HPP

#include <baseliner/stats/StatsRegistry.hpp>
#include <typeindex>
#include <vector>
namespace Baseliner::Stats {

  class IStatBase {
  public:
    virtual ~IStatBase() = default;

    // What types do i need
    [[nodiscard]] virtual auto dependencies() const -> std::vector<std::type_index> = 0;

    // What types do i provide
    [[nodiscard]] virtual auto output() const -> std::type_index = 0;

    virtual void compute(StatsRegistry &reg) = 0;
  };

  // Helper for Tag management.
  template <typename OutputTag, typename... InputTags>
  class IStat : public IStatBase {
  public:
    [[nodiscard]] auto dependencies() const -> std::vector<std::type_index> override {
      return {std::type_index(typeid(InputTags))...};
    }

    [[nodiscard]] auto output() const -> std::type_index override {
      return std::type_index(typeid(OutputTag));
    }
    virtual auto calculate(const typename InputTags::type &...inputs) -> typename OutputTag::type = 0;

    void compute(StatsRegistry &reg) final {
      auto result = calculate(reg.get<InputTags>()...);
      reg.set<OutputTag>(result);
    };
  };

} // namespace Baseliner::Stats

#endif // __BASELINER__ISTATS_HPP
