#pragma once

#include <type_traits>

#include "container_concepts.hpp"
#include "associative_container_concepts.hpp"
#include "map_container_concepts.hpp"


namespace core::meta::concepts {

template <typename C, typename ValueType>
concept container_of =
    container<C> and std::same_as<ValueType, typename C::value_type>;

template <typename C, typename ValueType>
concept mutable_container_of =
    container_of<C, ValueType> and mutable_container<C>;

template <typename C, typename ValueType>
concept sized_container_of = container_of<C, ValueType> and sized_container<C>;

template <typename C, typename ValueType>
concept clearable_container_of =
    container_of<C, ValueType> and clearable_container<C>;

template <typename C, typename ValueType>
concept reversible_container_of =
    container_of<C, ValueType> and reversible_container<C>;


     template <typename C, typename ValueType, typename KeyType = ValueType>
concept associative_container_of =
    container_of<C, ValueType> and associative_container<C> and
    std::same_as<KeyType, typename C::key_type>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unique_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept multiple_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    multiple_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    ordered_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    ordered_unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    ordered_multiple_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unordered_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    unordered_unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    unordered_multiple_associative_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept map_container_of =
    map_container<C> and std::same_as<KeyType, typename C::key_type> and
    std::same_as<MappedType, typename C::mapped_type>;

template <typename C, typename KeyType, typename MappedType>
concept unique_map_container_of =
    map_container_of<C, KeyType, MappedType> and unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept multiple_map_container_of =
    map_container_of<C, KeyType, MappedType> and multiple_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and ordered_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    ordered_unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    ordered_multiple_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and unordered_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    unordered_unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    unordered_multiple_map_container<C>;
}