#pragma once
#include <type_traits>
#include <utility>

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @struct convertible_to_anything
 * @brief Helper struct that can be implicitly converted to any type.
 * @tparam i Index parameter to differentiate between different instances.
 */
template <std::size_t i>
struct convertible_to_anything {
  /**
   * @brief Implicit conversion operator to any type T.
   * @tparam T The target type to convert to.
   */
  template <typename T>
  operator T(){};
};

/**
 * @concept is_constructable_with
 * @brief Concept to check if a type T can be constructed with I arguments.
 * @tparam T The type to check.
 * @tparam I The number of arguments to check for.
 */
template <typename T, std::size_t I>
concept is_constructable_with = requires {
  []<std::size_t... is>(std::index_sequence<is...> i_s)
      -> decltype(T{convertible_to_anything<is>{}...}) {
    return {};
  }(std::make_index_sequence<I>{});
};

/**
 * @concept aggregate_of
 * @brief Concept to check if a type T is an aggregate with exactly n members.
 * @tparam T The type to check.
 * @tparam n The number of members to check for.
 */
template <typename T, std::size_t n>
concept aggregate_of = std::is_aggregate_v<T> && is_constructable_with<T, n> &&
                       !is_constructable_with<T, n + 1>;

/**
 * @brief Maximum number of aggregate members to check.
 */
constexpr std::size_t maxAggregateMembers = 12;

/**
 * @var number_of_aggregate_members
 * @brief Variable template to determine the number of aggregate members in type
 * T.
 * @tparam T The type to check.
 */
template <typename T>
constexpr auto number_of_aggregate_members =
    []<std::size_t... indexes>(std::index_sequence<indexes...> i_s) {
      return ((aggregate_of<T, indexes> * indexes) + ... + 0);
    }(std::make_index_sequence<maxAggregateMembers>{});

}  // namespace core::meta::concepts
