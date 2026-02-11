#ifndef BASELINER_STATS_REGISTRY
#define BASELINER_STATS_REGISTRY
#include <any>
#include <typeindex>
#include <unordered_map>

namespace Baseliner::Stats {

  class StatsRegistry {
  public:
    // Writes data into the store.
    template <typename Tag>
    void set(typename Tag::type value) {
      m_storage[std::type_index(typeid(Tag))] = value;
    };

    // Reads data. Throws if missing or type mismatch.
    template <typename Tag>
    [[nodiscard]] auto get() const -> const typename Tag::type & {
      return std::any_cast<const typename Tag::type &>(m_storage.at(std::type_index(typeid(Tag))));
    };

    // Checks if a value exists (useful for optional outputs)
    template <typename Tag>
    [[nodiscard]] auto has() const -> bool {
      return m_storage.find(std::type_index(typeid(Tag))) != m_storage.end();
    };

  private:
    // Maps TypeIndex (Tag) -> std::any (Value)
    std::unordered_map<std::type_index, std::any> m_storage;
  };

} // namespace Baseliner::Stats
#endif // BASELINER_STATS_REGISTRY