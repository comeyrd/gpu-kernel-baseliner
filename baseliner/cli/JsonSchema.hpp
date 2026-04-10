#ifndef BASELINER_CLI_SCHEMA_GENERATOR_HPP
#define BASELINER_CLI_SCHEMA_GENERATOR_HPP

#include <baseliner/cli/Serializer.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Baseliner::Ser {

  template <typename T>
  struct extract_member_type;

  template <typename Class, typename Member>
  struct extract_member_type<Member Class::*> {
    using type = Member;
  };

  template <typename T>
  struct is_vector : std::false_type {};

  template <typename T, typename A>
  struct is_vector<std::vector<T, A>> : std::true_type {};

  template <typename T>
  struct is_unordered_map : std::false_type {};

  template <typename K, typename V, typename H, typename E, typename A>
  struct is_unordered_map<std::unordered_map<K, V, H, E, A>> : std::true_type {};

  template <typename T>
  nlohmann::ordered_json generate_raw_schema();

  template <typename T>
  nlohmann::ordered_json generate_raw_schema() {
    using DecayedT = std::decay_t<T>;
    nlohmann::ordered_json schema = nlohmann::ordered_json::object();

    if constexpr (std::is_same_v<DecayedT, std::string>) {
      schema["type"] = "string";
    } else if constexpr (std::is_same_v<DecayedT, bool>) {
      schema["type"] = "boolean";
    } else if constexpr (std::is_integral_v<DecayedT>) {
      schema["type"] = "integer";
    } else if constexpr (std::is_floating_point_v<DecayedT>) {
      schema["type"] = "number";
    } else if constexpr (is_optional<DecayedT>::value) {
      using InnerType = typename DecayedT::value_type;
      return generate_raw_schema<InnerType>();
    } else if constexpr (is_vector<DecayedT>::value) {
      using InnerType = typename DecayedT::value_type;
      schema["type"] = "array";
      schema["items"] = generate_raw_schema<InnerType>();
    } else if constexpr (is_unordered_map<DecayedT>::value) {
      using ValueType = typename DecayedT::mapped_type;
      schema["type"] = "object";
      schema["additionalProperties"] = generate_raw_schema<ValueType>();
    } else if constexpr (is_described_enum<DecayedT>::value) {
      schema["type"] = "string";
      nlohmann::ordered_json enum_list = nlohmann::ordered_json::array();

      std::apply([&](const auto &...values) { (..., [&]() { enum_list.push_back(values.key); }()); },
                 EnumTraits<DecayedT>::values());

      schema["enum"] = enum_list;
    } else if constexpr (is_described<DecayedT>::value) {
      schema["type"] = "object";
      nlohmann::ordered_json properties = nlohmann::ordered_json::object();
      nlohmann::ordered_json required = nlohmann::ordered_json::array();

      std::apply(
          [&](const auto &...fields) {
            (..., [&]() {
              using PtrType = decltype(fields.ptr);
              using FieldType = typename extract_member_type<PtrType>::type;

              properties[fields.key] = generate_raw_schema<FieldType>();

              if constexpr (!is_optional<FieldType>::value) {
                required.push_back(fields.key);
              }
            }());
          },
          SerTraits<DecayedT>::fields());

      schema["properties"] = properties;

      if (!required.empty()) {
        schema["required"] = required;
      }

      schema["additionalProperties"] = false;
    }

    return schema;
  }

  template <typename RootType>
  nlohmann::ordered_json export_json_schema(const std::string &title = "Baseliner Protocol Schema") {
    nlohmann::ordered_json schema = generate_raw_schema<RootType>();

    schema["$schema"] = "http://json-schema.org/draft-07/schema#";
    schema["title"] = title;

    return schema;
  }

} // namespace Baseliner::Ser

#endif // BASELINER_CLI_SCHEMA_GENERATOR_HPP