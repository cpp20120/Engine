#pragma once

#include <type_traits>

#include "associative_container_concepts.hpp"
#include "container_concepts.hpp"
#include "map_container_concepts.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept container_of
 * @brief Concept to check if a type C is a container of ValueType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 */
template <typename C, typename ValueType>
concept container_of =
    container<C> and std::same_as<ValueType, typename C::value_type>;

/**
 * @concept mutable_container_of
 * @brief Concept to check if a type C is a mutable container of ValueType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 */
template <typename C, typename ValueType>
concept mutable_container_of =
    container_of<C, ValueType> and mutable_container<C>;

/**
 * @concept sized_container_of
 * @brief Concept to check if a type C is a sized container of ValueType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 */
template <typename C, typename ValueType>
concept sized_container_of = container_of<C, ValueType> and sized_container<C>;

/**
 * @concept clearable_container_of
 * @brief Concept to check if a type C is a clearable container of ValueType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 */
template <typename C, typename ValueType>
concept clearable_container_of =
    container_of<C, ValueType> and clearable_container<C>;

/**
 * @concept reversible_container_of
 * @brief Concept to check if a type C is a reversible container of ValueType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 */
template <typename C, typename ValueType>
concept reversible_container_of =
    container_of<C, ValueType> and reversible_container<C>;

/**
 * @concept associative_container_of
 * @brief Concept to check if a type C is an associative container of ValueType
 * with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept associative_container_of =
    container_of<C, ValueType> and associative_container<C> and
    std::same_as<KeyType, typename C::key_type>;

/**
 * @concept unique_associative_container_of
 * @brief Concept to check if a type C is a unique associative container of
 * ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept unique_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unique_associative_container<C>;

/**
 * @concept multiple_associative_container_of
 * @brief Concept to check if a type C is a multiple associative container of
 * ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept multiple_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    multiple_associative_container<C>;

/**
 * @concept ordered_associative_container_of
 * @brief Concept to check if a type C is an ordered associative container of
 * ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    ordered_associative_container<C>;

/**
 * @concept ordered_unique_associative_container_of
 * @brief Concept to check if a type C is an ordered unique associative
 * container of ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    ordered_unique_associative_container<C>;

/**
 * @concept ordered_multiple_associative_container_of
 * @brief Concept to check if a type C is an ordered multiple associative
 * container of ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    ordered_multiple_associative_container<C>;

/**
 * @concept unordered_associative_container_of
 * @brief Concept to check if a type C is an unordered associative container of
 * ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unordered_associative_container<C>;

/**
 * @concept unordered_unique_associative_container_of
 * @brief Concept to check if a type C is an unordered unique associative
 * container of ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    unordered_unique_associative_container<C>;

/**
 * @concept unordered_multiple_associative_container_of
 * @brief Concept to check if a type C is an unordered multiple associative
 * container of ValueType with KeyType.
 * @tparam C The container type to check.
 * @tparam ValueType The value type the container should hold.
 * @tparam KeyType The key type the container should use.
 */
template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    unordered_multiple_associative_container<C>;

/**
 * @concept map_container_of
 * @brief Concept to check if a type C is a map container with KeyType and
 * MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept map_container_of =
    map_container<C> and std::same_as<KeyType, typename C::key_type> and
    std::same_as<MappedType, typename C::mapped_type>;

/**
 * @concept unique_map_container_of
 * @brief Concept to check if a type C is a unique map container with KeyType
 * and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept unique_map_container_of =
    map_container_of<C, KeyType, MappedType> and unique_map_container<C>;

/**
 * @concept multiple_map_container_of
 * @brief Concept to check if a type C is a multiple map container with KeyType
 * and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept multiple_map_container_of =
    map_container_of<C, KeyType, MappedType> and multiple_map_container<C>;

/**
 * @concept ordered_map_container_of
 * @brief Concept to check if a type C is an ordered map container with KeyType
 * and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept ordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and ordered_map_container<C>;

/**
 * @concept ordered_unique_map_container_of
 * @brief Concept to check if a type C is an ordered unique map container with
 * KeyType and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept ordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    ordered_unique_map_container<C>;

/**
 * @concept ordered_multiple_map_container_of
 * @brief Concept to check if a type C is an ordered multiple map container with
 * KeyType and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept ordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    ordered_multiple_map_container<C>;

/**
 * @concept unordered_map_container_of
 * @brief Concept to check if a type C is an unordered map container with
 * KeyType and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept unordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and unordered_map_container<C>;

/**
 * @concept unordered_unique_map_container_of
 * @brief Concept to check if a type C is an unordered unique map container with
 * KeyType and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept unordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    unordered_unique_map_container<C>;

/**
 * @concept unordered_multiple_map_container_of
 * @brief Concept to check if a type C is an unordered multiple map container
 * with KeyType and MappedType.
 * @tparam C The container type to check.
 * @tparam KeyType The key type the container should use.
 * @tparam MappedType The mapped type the container should hold.
 */
template <typename C, typename KeyType, typename MappedType>
concept unordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    unordered_multiple_map_container<C>;
}  // namespace core::meta::concepts
