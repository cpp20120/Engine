#pragma once

#include <concepts>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace core::meta::type_builder {

/**
 * @brief Concept to check if a type can be constructed from std::string.
 */
template <typename T>
concept StringLike = std::constructible_from<std::string, T>;

/**
 * @brief Concept to check if a type is an integral (e.g., int, size_t).
 */
template <typename T>
concept IntegralId = std::integral<std::remove_cvref_t<T>>;

/**
 * @brief Concept to check if a type is printable via std::ostream.
 */
template <typename T>
concept Printable = requires(std::ostream& os, const T& val) { os << val; };

/**
 * @brief Concept that ensures all provided types are Printable.
 */
template <typename... Ts>
concept ValidMetadata = (Printable<Ts> && ...);

/**
 * @brief Concept that defines a valid custom type composed of:
 * a StringLike name, an IntegralId, and optional Printable metadata.
 */
template <typename NameType, typename IdType, typename... MetadataTypes>
concept ValidCustomType = StringLike<NameType> && IntegralId<IdType> &&
                          ValidMetadata<MetadataTypes...>;

/**
 * @brief A builder class for defining a custom type with a name, id, and
 * optional metadata.
 *
 * @tparam NameType The type used for the name (must satisfy StringLike).
 * @tparam IdType The type used for the ID (must satisfy IntegralId).
 * @tparam MetadataTypes Variadic list of metadata types (must all satisfy
 * Printable).
 */
template <typename NameType = std::string, typename IdType = int,
          typename... MetadataTypes>
  requires ValidCustomType<NameType, IdType, MetadataTypes...>
class CustomTypeBuilder {
 public:
  NameType name;                          ///< The name of the custom type.
  IdType id;                              ///< The ID of the custom type.
  std::tuple<MetadataTypes...> metadata;  ///< The associated metadata.

  /**
   * @brief Constructs a CustomTypeBuilder from provided name, id, and metadata.
   *
   * @param name The name value.
   * @param id The id value.
   * @param metadata The metadata values.
   */
  constexpr CustomTypeBuilder(
      NameType&& name, IdType id,
      MetadataTypes&&... metadata) noexcept(std::
                                                is_nothrow_constructible_v<
                                                    NameType, NameType&&> &&
                                            (std::is_nothrow_constructible_v<
                                                 MetadataTypes,
                                                 MetadataTypes&&> &&
                                             ...))
      : name(std::forward<NameType>(name)),
        id(id),
        metadata(std::forward<MetadataTypes>(metadata)...) {}

  /**
   * @brief Creates a new builder instance with a different name.
   *
   * @tparam NewNameType The new type for the name (must satisfy StringLike).
   * @param new_name The new name value.
   * @return A new CustomTypeBuilder with updated name.
   */
  template <StringLike NewNameType>
  constexpr auto with_name(NewNameType&& new_name) const& {
    return CustomTypeBuilder<NewNameType, IdType, MetadataTypes...>(
        std::forward<NewNameType>(new_name), id, metadata);
  }

  /**
   * @brief Creates a new builder instance with a different ID.
   *
   * @tparam NewIdType The new type for the ID (must satisfy IntegralId).
   * @param new_id The new ID value.
   * @return A new CustomTypeBuilder with updated ID.
   */
  template <IntegralId NewIdType>
  constexpr auto with_id(NewIdType new_id) const& {
    return CustomTypeBuilder<NameType, NewIdType, MetadataTypes...>(
        name, new_id, metadata);
  }

  /**
   * @brief Creates a new builder instance by appending new metadata.
   *
   * @tparam NewMetadata The new metadata type (must be Printable).
   * @param data The new metadata value.
   * @return A new CustomTypeBuilder with the added metadata.
   */
  template <Printable NewMetadata>
  constexpr auto with_metadata(NewMetadata&& data) const& {
    return std::apply(
        [&](auto&&... args) {
          return CustomTypeBuilder<NameType, IdType, MetadataTypes...,
                                   NewMetadata>(
              name, id, std::forward<decltype(args)>(args)...,
              std::forward<NewMetadata>(data));
        },
        metadata);
  }

  /**
   * @brief Displays the contents of the builder to std::cout.
   * Requires NameType and IdType to be Printable.
   */
  constexpr void display() const
    requires(Printable<NameType> && Printable<IdType>)
  {
    std::cout << "Name: " << name << "\nID: " << id;
    std::apply(
        [](const auto&... args) {
          (..., (std::cout << "\nMetadata: " << args));
        },
        metadata);
    std::cout << "\n";
  }

  /**
   * @brief Converts the builder contents into a JSON-like string.
   * Requires NameType and IdType to be Printable.
   *
   * @return std::string JSON-formatted representation of the builder.
   */
  constexpr std::string to_json() const
    requires(Printable<NameType> && Printable<IdType>)
  {
    std::string result = "{\n  \"name\": \"" + std::string(name) +
                         "\",\n  \"id\": " + std::to_string(id);
    std::apply(
        [&result](const auto&... args) {
          (..., (result += ",\n  \"metadata\": \"" + std::string(args) + "\""));
        },
        metadata);
    return result + "\n}";
  }
};

/**
 * @brief Factory function to create a CustomTypeBuilder without metadata.
 *
 * @tparam NameType The type used for the name.
 * @tparam IdType The type used for the ID.
 * @param name The name value.
 * @param id The id value.
 * @return CustomTypeBuilder instance.
 */
template <StringLike NameType, IntegralId IdType>
constexpr auto create_builder(NameType&& name, IdType id) {
  return CustomTypeBuilder<std::remove_cvref_t<NameType>,
                           std::remove_cvref_t<IdType>>(
      std::forward<NameType>(name), id);
}

/**
 * @brief Factory function to create a CustomTypeBuilder with metadata.
 *
 * @tparam NameType The type used for the name.
 * @tparam IdType The type used for the ID.
 * @tparam MetadataTypes The types used for the metadata.
 * @param name The name value.
 * @param id The id value.
 * @param metadata The metadata values.
 * @return CustomTypeBuilder instance.
 */
template <StringLike NameType, IntegralId IdType, Printable... MetadataTypes>
constexpr auto create_builder(NameType&& name, IdType id,
                              MetadataTypes&&... metadata) {
  return CustomTypeBuilder<std::remove_cvref_t<NameType>,
                           std::remove_cvref_t<IdType>,
                           std::remove_cvref_t<MetadataTypes>...>(
      std::forward<NameType>(name), id,
      std::forward<MetadataTypes>(metadata)...);
}

}  // namespace core::meta::type_builder
