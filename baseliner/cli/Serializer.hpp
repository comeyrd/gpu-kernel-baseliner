#ifndef BASELINER_CLI_Serializer_HPP
#define BASELINER_CLI_Serializer_HPP

#include <fstream>
#include <optional>
#include <tuple>

namespace Baseliner::Ser {

  // ── Descriptors ───────────────────────────────────────────────────────────────

  template <typename Owner, typename Field>
  struct FieldDescriptor {
    const char *key;
    Field Owner::*ptr;
  };

  template <typename E>
  struct EnumDescriptor {
    const char *key;
    E value;
  };

  // ── Traits ────────────────────────────────────────────────────────────────────

  template <typename T>
  struct SerTraits {};
  template <typename T>
  struct EnumTraits {};

  // ── Detection ─────────────────────────────────────────────────────────────────

  template <typename T, typename = void>
  struct is_described : std::false_type {};

  template <typename T>
  struct is_described<T, std::void_t<decltype(SerTraits<T>::fields())>> : std::true_type {};

  template <typename T, typename = void>
  struct is_described_enum : std::false_type {};

  template <typename T>
  struct is_described_enum<T, std::void_t<decltype(EnumTraits<T>::values())>> : std::true_type {};

  template <typename T>
  struct is_optional : std::false_type {};

  template <typename T>
  struct is_optional<std::optional<T>> : std::true_type {};

  // ── Entry points ──────────────────────────────────────────────────────────────

  template <typename Backend, typename T>
  void serialize(std::ostream &oss, const T &obj) {
    Backend::serialize(oss, obj);
  }

  template <typename Backend, typename T>
  void deserialize(std::istream &iss, T &obj) {
    Backend::deserialize(iss, obj);
  }

  template <typename Backend, typename T>
  auto from_file(const std::string &filename) -> T {
    T object;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
      // throw Errors::file_read_error(filename);
      throw std::runtime_error("File reading error");
    }
    Ser::deserialize<Backend>(infile, object);
    return object;
  }

  template <typename Backend, typename T>
  void to_file(const T &object, const std::string &filename) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
      // throw Errors::file_write_error(filename);
      throw std::runtime_error("File reading error");
    }
    Ser::serialize<Backend>(file, object);
  }

} // namespace Baseliner::Ser

// ── Macros ────────────────────────────────────────────────────────────────────

#define FIELD(member)                                                                                                  \
  ::Baseliner::Ser::FieldDescriptor<Self, decltype(Self::member)> {                                                    \
    #member, &Self::member                                                                                             \
  }

#define DESCRIBE(Type, ...)                                                                                            \
  namespace Ser {                                                                                                      \
    template <>                                                                                                        \
    struct SerTraits<Type> {                                                                                           \
      using Self = Type;                                                                                               \
      static constexpr auto fields() {                                                                                 \
        return std::make_tuple(__VA_ARGS__);                                                                           \
      }                                                                                                                \
    };                                                                                                                 \
  }

#define ENUM_VALUE(value)                                                                                              \
  ::Baseliner::Ser::EnumDescriptor<Self> {                                                                             \
    #value, Self::value                                                                                                \
  }

#define DESCRIBE_ENUM(Type, ...)                                                                                       \
  namespace Ser {                                                                                                      \
    template <>                                                                                                        \
    struct EnumTraits<Type> {                                                                                          \
      using Self = Type;                                                                                               \
      static constexpr auto values() {                                                                                 \
        return std::make_tuple(__VA_ARGS__);                                                                           \
      }                                                                                                                \
    };                                                                                                                 \
  }

#define ENUM_VALUE(value)                                                                                              \
  ::Baseliner::Ser::EnumDescriptor<Self> {                                                                             \
    #value, Self::value                                                                                                \
  }

#define DESCRIBE_TEMPLATE(Type, ...)                                                                                   \
  namespace Ser {                                                                                                      \
    template <typename T_>                                                                                             \
    struct SerTraits<Type<T_>> {                                                                                       \
      using Self = Type<T_>;                                                                                           \
      static constexpr auto fields() {                                                                                 \
        return std::make_tuple(__VA_ARGS__);                                                                           \
      }                                                                                                                \
    };                                                                                                                 \
  }

#endif // BASELINER_CLI_SERIALIZER_HPP