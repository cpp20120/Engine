#pragma once
#include <type_traits>

#include "../traits/type_traits.hpp"
#include "./optional_concepts.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept is_converting_ctor_v
 * @brief Concept to check if type T has a converting constructor for type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_converting_ctor_v =
    core::meta::type_traits::is_converting_ctor<T, U>::value;

/**
 * @concept is_converting_assign_v
 * @brief Concept to check if type T has a converting assignment operator for
 * type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_converting_assign_v =
    core::meta::type_traits::is_converting_assign<T, U>::value;

/**
 * @concept is_constructible_from_v
 * @brief Concept to check if type T is constructible from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_constructible_from_v = std::is_constructible_v<T, U>;

/**
 * @concept is_assignable_from
 * @brief Concept to check if type T is assignable from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_assignable_from = std::is_assignable_v<T&, U>;

/**
 * @concept is_implicitly_convertible
 * @brief Concept to check if type U is implicitly convertible to type T.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_implicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor_v<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_convertible_v<const U&, T>;

/**
 * @concept is_explicitly_convertible
 * @brief Concept to check if type U is explicitly convertible to type T.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_explicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor_v<T, U> &&
    std::is_constructible_v<T, const U&> && !std::is_convertible_v<const U&, T>;

/**
 * @concept is_implicitly_move_convertible
 * @brief Concept to check if type U is implicitly move convertible to type T.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_implicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor_v<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

/**
 * @concept is_explicitly_move_convertible
 * @brief Concept to check if type U is explicitly move convertible to type T.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_explicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor_v<T, U> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

/**
 * @concept is_directly_constructible
 * @brief Concept to check if type T is directly constructible from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_directly_constructible =
    !core::meta::type_traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

/**
 * @concept is_explicitly_directly_constructible
 * @brief Concept to check if type T is explicitly directly constructible from
 * type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_explicitly_directly_constructible =
    !core::meta::type_traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

/**
 * @concept is_copy_assignable_from
 * @brief Concept to check if type T is copy assignable from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_copy_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign_v<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

/**
 * @concept is_move_assignable_from
 * @brief Concept to check if type T is move assignable from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_move_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign_v<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_assignable_v<T&, U&&>;

}  // namespace core::meta::concepts
