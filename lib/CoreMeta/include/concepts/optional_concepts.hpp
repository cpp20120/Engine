#pragma once

/**
 * @file optional_concepts.hpp
 * @brief Optional concepts for handling optional-like types.
 */

#include <type_traits>

#include "../traits/type_traits.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept convertible_to_optional_like
 * @brief Concept to check if type T is convertible to an optional-like type U.
 * @tparam T The source type.
 * @tparam U The target optional-like type.
 */
template <typename T, typename U>
concept convertible_to_optional_like =
    core::meta::type_traits::is_convertible_to_optional_like<T, U>::value;

/**
 * @concept assingable_from_optional_like
 * @brief Concept to check if type T is assignable from an optional-like type U.
 * @tparam T The target type.
 * @tparam U The source optional-like type.
 */
template <typename T, typename U>
concept assingable_from_optional_like =
    core::meta::type_traits::is_assignable_from_optional_like<T, U>::value;

/**
 * @concept optional_like
 * @brief Concept to check if type T is an optional-like type.
 * @tparam T The type to check.
 */
template <typename T>
concept optional_like = core::meta::type_traits::is_optional_like<T>::value;

/**
 * @concept is_converting_ctor
 * @brief Concept to check if type T has a converting constructor from type U,
 * excluding references and optional types.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_converting_ctor =
    !std::disjunction_v<std::is_reference<U>,
                        std::is_constructible<T, std::optional<U>&>,
                        std::is_constructible<T, const std::optional<U>&>,
                        std::is_constructible<T, std::optional<U>&&>,
                        std::is_constructible<T, const std::optional<U>&&>,
                        std::is_convertible<std::optional<U>&, T>,
                        std::is_convertible<const std::optional<U>&, T>,
                        std::is_convertible<std::optional<U>&&, T>,
                        std::is_convertible<const std::optional<U>&&, T>>;

/**
 * @concept assignable_value
 * @brief Concept to check if type T is assignable from type U, excluding
 * optional-like types and scalar types.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept assignable_value =
    !core::meta::type_traits::is_optional_like<std::decay_t<U>>::value &&
    std::is_constructible_v<T, U> &&
    core::meta::type_traits::is_assignable_from_optional_like<T, U>::value &&
    !std::conjunction_v<std::is_scalar<T>, std::is_same<T, std::decay_t<U>>>;

/**
 * @concept moveable_assign_from
 * @brief Concept to check if type T is move assignable from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept moveable_assign_from =
    !std::is_same_v<U, T> &&
    core::meta::type_traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, U> && std::is_assignable_v<T&, U>;

/**
 * @concept copyable_assign_from
 * @brief Concept to check if type T is copy assignable from type U.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept copyable_assign_from =
    !std::is_same_v<U, T> &&
    core::meta::type_traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

/**
 * @concept is_converting_assign
 * @brief Concept to check if type T is converting assignable from type U,
 * excluding optional-like types.
 * @tparam T The target type.
 * @tparam U The source type.
 */
template <typename T, typename U>
concept is_converting_assign =
    is_converting_ctor<T, U> &&
    !std::disjunction_v<std::is_assignable<T&, std::optional<U>&>,
                        std::is_assignable<T&, const std::optional<U>&>,
                        std::is_assignable<T&, std::optional<U>&&>,
                        std::is_assignable<T&, const std::optional<U>&&>>;

}  // namespace core::meta::concepts
