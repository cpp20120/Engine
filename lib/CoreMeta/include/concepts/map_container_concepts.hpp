#pragma once

#include "associative_container_concepts.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept Emplaceable
 * @brief Concept to check if a type C supports emplace operations with given
 * arguments.
 * @tparam C The container type to check.
 * @tparam Key The key type to emplace.
 * @tparam Args The argument types to emplace.
 */
template <typename C, typename Key, typename... Args>
concept Emplaceable = requires(C& cont, typename C::const_iterator const& hint,
                               Key&& key, Args&&... args) {
  {
    cont.emplace_hint(hint, std::forward<Key>(key), std::forward<Args>(args)...)
  } -> std::same_as<typename C::iterator>;
  {
    cont.emplace_hint(hint, std::piecewise_construct,
                      std::forward_as_tuple(std::forward<Key>(key)),
                      std::forward_as_tuple(std::forward<Args>(args)...))
  } -> std::same_as<typename C::iterator>;
};

/**
 * @concept InsertOrAssignable
 * @brief Concept to check if a type C supports insert or assign operations with
 * given key and value.
 * @tparam C The container type to check.
 * @tparam Key The key type to insert or assign.
 * @tparam Value The value type to insert or assign.
 */
template <typename C, typename Key, typename Value>
concept InsertOrAssignable = requires(
    C& cont, typename C::const_iterator const& hint, Key&& key, Value&& value) {
  {
    cont.insert_or_assign(std::forward<Key>(key), std::forward<Value>(value))
  } -> std::same_as<std::pair<typename C::iterator, bool>>;
  {
    cont.insert_or_assign(hint, std::forward<Key>(key),
                          std::forward<Value>(value))
  } -> std::same_as<typename C::iterator>;
};

/**
 * @concept TryEmplaceable
 * @brief Concept to check if a type C supports try emplace operations with
 * given key and arguments.
 * @tparam C The container type to check.
 * @tparam Key The key type to try emplace.
 * @tparam Args The argument types to try emplace.
 */
template <typename C, typename Key, typename... Args>
concept TryEmplaceable = requires(C& cont,
                                  typename C::const_iterator const& hint,
                                  Key&& key, Args&&... args) {
  {
    cont.try_emplace(std::forward<Key>(key), std::forward<Args>(args)...)
  } -> std::same_as<std::pair<typename C::iterator, bool>>;
  {
    cont.try_emplace(hint, std::forward<Key>(key), std::forward<Args>(args)...)
  } -> std::same_as<typename C::iterator>;
};

/**
 * @concept map_container
 * @brief Concept to check if a type C is a map container.
 * @tparam C The type to check.
 */
template <typename C>
concept map_container = associative_container<C> and requires(C& cont) {
  typename C::mapped_type;
  requires requires(typename C::const_iterator const& hint) {
    requires Emplaceable<C, typename C::key_type> or requires {
      requires not std::copyable<typename C::key_type> or
                   Emplaceable<C, typename C::key_type const&>;
      requires not std::movable<typename C::key_type> or
                   Emplaceable<C, typename C::key_type&&>;
    };
  };
};

/**
 * @concept unique_map_container
 * @brief Concept to check if a type C is a unique map container.
 * @tparam C The type to check.
 */
template <typename C>
concept unique_map_container = map_container<C> and requires(
                                                        C& cont,
                                                        C const& const_cont) {
  requires requires(typename C::key_type const& key,
                    typename C::key_type&& tmp_key) {
    requires not std::default_initializable<typename C::mapped_type> or
                 requires {
                   requires not std::copyable<typename C::key_type> or
                                requires {
                                  {
                                    cont[key]
                                  } -> std::same_as<typename C::mapped_type&>;
                                };
                   requires not std::movable<typename C::key_type> or requires {
                     {
                       cont[std::move(tmp_key)]
                     } -> std::same_as<typename C::mapped_type&>;
                   };
                 };
    { cont.at(key) } -> std::same_as<typename C::mapped_type&>;
    { const_cont.at(key) } -> std::same_as<typename C::mapped_type const&>;
  };

  requires requires(typename C::const_iterator const& hint) {
    requires InsertOrAssignable<C, typename C::key_type,
                                typename C::mapped_type> or
                 requires {
                   requires not std::copyable<typename C::key_type> or
                                InsertOrAssignable<C,
                                                   typename C::key_type const&,
                                                   typename C::mapped_type>;
                   requires not std::movable<typename C::key_type> or
                                InsertOrAssignable<C, typename C::key_type&&,
                                                   typename C::mapped_type>;
                 };
  };
};

/**
 * @concept multiple_map_container
 * @brief Concept to check if a type C is a multiple map container.
 * @tparam C The type to check.
 */
template <typename C>
concept multiple_map_container = map_container<C> and requires(C& cont) {
  requires requires(typename C::const_iterator const& hint) {
    requires Emplaceable<C, typename C::key_type> or requires {
      requires not std::copyable<typename C::key_type> or
                   Emplaceable<C, typename C::key_type const&>;
      requires not std::movable<typename C::key_type> or
                   Emplaceable<C, typename C::key_type&&>;
    };
  };
};

/**
 * @concept ordered_map_container
 * @brief Concept to check if a type C is an ordered map container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_map_container =
    map_container<C> and ordered_associative_container<C>;

/**
 * @concept ordered_unique_map_container
 * @brief Concept to check if a type C is an ordered unique map container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_unique_map_container =
    unique_map_container<C> and ordered_map_container<C>;

/**
 * @concept ordered_multiple_map_container
 * @brief Concept to check if a type C is an ordered multiple map container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_multiple_map_container =
    multiple_map_container<C> and ordered_map_container<C>;

/**
 * @concept unordered_map_container
 * @brief Concept to check if a type C is an unordered map container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_map_container =
    map_container<C> and unordered_associative_container<C>;

/**
 * @concept unordered_unique_map_container
 * @brief Concept to check if a type C is an unordered unique map container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_unique_map_container =
    unique_map_container<C> and unordered_map_container<C>;

/**
 * @concept unordered_multiple_map_container
 * @brief Concept to check if a type C is an unordered multiple map container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_multiple_map_container =
    multiple_map_container<C> and unordered_map_container<C>;

}  // namespace core::meta::concepts
