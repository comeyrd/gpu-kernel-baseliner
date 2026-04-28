#ifndef BASELINER_CLI_JSON_HPP
#define BASELINER_CLI_JSON_HPP

#include <baseliner/cli/Serializer.hpp>
#include <baseliner/core/Durations.hpp>

#include <baseliner/core/Metric.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

NLOHMANN_JSON_NAMESPACE_BEGIN
template <>
struct adl_serializer<Baseliner::float_milliseconds> {
  static void to_json(nlohmann::ordered_json &json_obj, const Baseliner::float_milliseconds &value) {
    json_obj = value.count();
  }

  static void from_json(const nlohmann::ordered_json &json_obj, Baseliner::float_milliseconds &value) {
    value = Baseliner::float_milliseconds(json_obj.get<float>());
  }
};
template <>
struct adl_serializer<std::monostate> {
  static void to_json(ordered_json &json_obj, const std::monostate & /*monostate*/) {
    json_obj = "";
  }
};
template <typename... Ts>
struct adl_serializer<std::variant<Ts...>> {
  static void to_json(nlohmann::ordered_json &json_obj, const std::variant<Ts...> &var_val) {
    std::visit([&](const auto &val) { json_obj = val; }, var_val);
  }

  static void from_json(const nlohmann::ordered_json & /*json*/, std::variant<Ts...> & /*variant*/) {
    throw std::runtime_error("variant deserialization not supported");
  }
};
NLOHMANN_JSON_NAMESPACE_END

namespace Baseliner {

  // ── to_json ───────────────────────────────────────────────────────────────────

  template <typename T>
  auto to_json(nlohmann::ordered_json &json, const T &obj) ->
      typename std::enable_if<Ser::is_described<T>::value>::type {
    std::apply(
        [&](const auto &...fields) {
          (
              [&] {
                const auto &value = obj.*(fields.ptr);
                using FieldType = std::remove_cv_t<std::remove_reference_t<decltype(value)>>;

                if constexpr (Ser::is_optional<FieldType>::value) {
                  if (value.has_value()) {
                    json[fields.key] = *value;
                  }
                } else {
                  json[fields.key] = value;
                }
              }(),
              ...);
        },
        Ser::SerTraits<T>::fields());
  }

  template <typename T>
  auto to_json(nlohmann::ordered_json &json, const T &value) ->
      typename std::enable_if<Ser::is_described_enum<T>::value>::type {
    bool found = false;
    std::apply(
        [&](const auto &...values) {
          (
              [&] {
                if (!found && value == values.value) {
                  json = values.key;
                  found = true;
                }
              }(),
              ...);
        },
        Ser::EnumTraits<T>::values());
  }

  // ── Context Interceptors ──────────────────────────────────────────────────────

  inline void throw_enriched_error(const std::string &path, const nlohmann::ordered_json &node,
                                   const std::exception &e) {
    std::string msg = e.what();
    if (msg.find("[Baseliner JSON Error]") == 0) {
      throw std::runtime_error(msg);
    }

    std::string snippet;
    try {
      snippet = node.dump(2);
      if (snippet.length() > 200) {
        snippet = snippet.substr(0, 197) + "...";
      }
    } catch (...) {
      snippet = "<unprintable>";
    }

    throw std::runtime_error("[Baseliner JSON Error] at path '" + path + "':\nReason: " + msg + "\nJSON snippet:\n" +
                             snippet);
  }

  template <typename T>
  auto from_json(const nlohmann::ordered_json &json, T &obj, const std::string &path) ->
      typename std::enable_if<Ser::is_described<T>::value>::type;

  template <typename T>
  auto from_json(const nlohmann::ordered_json &json, T &value) ->
      typename std::enable_if<Ser::is_described_enum<T>::value>::type;

  template <typename T>
  void extract_with_context(const nlohmann::ordered_json &j, T &val, const std::string &path) {
    try {
      j.get_to(val);
    } catch (const std::exception &e) {
      throw_enriched_error(path, j, e);
    }
  }

  template <typename K, typename V, typename Hash, typename Eq, typename Alloc>
  void extract_with_context(const nlohmann::ordered_json &j, std::unordered_map<K, V, Hash, Eq, Alloc> &map,
                            const std::string &path) {
    if (!j.is_object())
      throw_enriched_error(path, j, std::runtime_error("Expected object {...}"));
    map.clear();
    for (const auto &[key, value] : j.items()) {
      std::string item_path = path.empty() ? key : path + "." + key;
      try {
        V item;
        if constexpr (Ser::is_described<V>::value) {
          from_json(value, item, item_path);
        } else if constexpr (Ser::is_described_enum<V>::value) {
          from_json(value, item);
        } else {
          extract_with_context(value, item, item_path);
        }
        map.emplace(key, std::move(item));
      } catch (const std::exception &e) {
        throw_enriched_error(item_path, value, e);
      }
    }
  }

  template <typename T, typename Alloc>
  void extract_with_context(const nlohmann::ordered_json &j, std::vector<T, Alloc> &vec, const std::string &path) {
    if (!j.is_array())
      throw_enriched_error(path, j, std::runtime_error("Expected array [...]"));
    vec.clear();
    vec.reserve(j.size());
    size_t index = 0;
    for (const auto &value : j) {
      std::string item_path = path + "[" + std::to_string(index) + "]";
      try {
        T item;
        if constexpr (Ser::is_described<T>::value) {
          from_json(value, item, item_path);
        } else if constexpr (Ser::is_described_enum<T>::value) {
          from_json(value, item);
        } else {
          extract_with_context(value, item, item_path);
        }
        vec.push_back(std::move(item));
      } catch (const std::exception &e) {
        throw_enriched_error(item_path, value, e);
      }
      index++;
    }
  }

  // ── from_json ─────────────────────────────────────────────────────────────────

  template <typename T>
  auto from_json(const nlohmann::ordered_json &json, T &obj, const std::string &path) ->
      typename std::enable_if<Ser::is_described<T>::value>::type {
    std::apply(
        [&](const auto &...fields) {
          (
              [&] {
                using FieldType = std::remove_cv_t<std::remove_reference_t<decltype(obj.*(fields.ptr))>>;
                std::string field_path = path.empty() ? fields.key : path + "." + fields.key;

                if (!json.contains(fields.key)) {
                  if constexpr (!Ser::is_optional<FieldType>::value) {
                    throw std::runtime_error("Missing required field: " + std::string(fields.key));
                  }
                  return;
                }

                auto &value = obj.*(fields.ptr);

                if constexpr (Ser::is_optional<FieldType>::value) {
                  if (json.at(fields.key).is_null()) {
                    value = std::nullopt;
                  } else {
                    using Inner = typename FieldType::value_type;
                    Inner tmp;
                    if constexpr (Ser::is_described<Inner>::value) {
                      from_json(json.at(fields.key), tmp, field_path);
                    } else if constexpr (Ser::is_described_enum<Inner>::value) {
                      try {
                        from_json(json.at(fields.key), tmp);
                      } catch (const std::exception &e) {
                        throw_enriched_error(field_path, json.at(fields.key), e);
                      }
                    } else {
                      extract_with_context(json.at(fields.key), tmp, field_path);
                    }
                    value = std::move(tmp);
                  }
                } else {
                  if constexpr (Ser::is_described<FieldType>::value) {
                    from_json(json.at(fields.key), value, field_path);
                  } else if constexpr (Ser::is_described_enum<FieldType>::value) {
                    try {
                      from_json(json.at(fields.key), value);
                    } catch (const std::exception &e) {
                      throw_enriched_error(field_path, json.at(fields.key), e);
                    }
                  } else {
                    extract_with_context(json.at(fields.key), value, field_path);
                  }
                }
              }(),
              ...);
        },
        Ser::SerTraits<T>::fields());
  }

  template <typename T>
  auto from_json(const nlohmann::ordered_json &json, T &obj) ->
      typename std::enable_if<Ser::is_described<T>::value>::type {
    from_json(json, obj, "");
  }

  template <typename T>
  auto from_json(const nlohmann::ordered_json &json, T &value) ->
      typename std::enable_if<Ser::is_described_enum<T>::value>::type {
    std::string str = json.get<std::string>();
    bool found = false;
    std::apply(
        [&](const auto &...values) {
          (
              [&] {
                if (!found && str == values.key) {
                  value = values.value;
                  found = true;
                }
              }(),
              ...);
        },
        Ser::EnumTraits<T>::values());
    if (!found) {
      throw std::runtime_error("unknown enum value: " + str);
    }
  }
  // ── Backend tag ───────────────────────────────────────────────────────────────

  struct Json {
    template <typename T>
    static void serialize(std::ostream &oss, const T &obj) {
      nlohmann::ordered_json json_o = obj;
      oss << json_o.dump(2);
    }

    template <typename T>
    static void deserialize(std::istream &iss, T &obj) {
      nlohmann::ordered_json json_o;
      iss >> json_o;
      Baseliner::from_json(json_o, obj);
    }
  };

  // ── Global API ───────────────────────────────────────────────────────────────

  template <typename T, typename Backend = Json>
  auto from_file(const std::string &filename) -> T {
    return Ser::from_file<Backend, T>(filename);
  }

  template <typename T, typename Backend = Json>
  void to_file(const T &object, const std::string &filename) {
    Ser::to_file<Backend, T>(object, filename);
  }

  template <typename Backend = Json, typename T>
  void serialize(std::ostream &oss, const T &obj) {
    Backend::serialize(oss, obj);
  }

  template <typename Backend = Json, typename T>
  void deserialize(std::istream &iss, T &obj) {
    Backend::deserialize(iss, obj);
  }

} // namespace Baseliner

NLOHMANN_JSON_NAMESPACE_BEGIN

// This Bridge solves the Namespace/ADL issue once and for all
template <typename T>
struct adl_serializer<
    T, std::enable_if_t<::Baseliner::Ser::is_described<T>::value || ::Baseliner::Ser::is_described_enum<T>::value>> {
  static void to_json(nlohmann::ordered_json &json, const T &obj) {
    ::Baseliner::to_json(json, obj);
  }

  static void from_json(const nlohmann::ordered_json &json, T &obj) {
    ::Baseliner::from_json(json, obj);
  }
};

NLOHMANN_JSON_NAMESPACE_END

#endif