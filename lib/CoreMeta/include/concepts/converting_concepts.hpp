#pragma once
#include <type_traits>
#include "../traits/type_traits.hpp"

namespace core::meta::concepts {

template <typename T, typename U>
concept is_converting_ctor_v = utils::traits::is_converting_ctor<T, U>::value;

template <typename T, typename U>
concept is_converting_assign_v =
    utils::traits::is_converting_assign<T, U>::value;

template <typename T, typename U>
concept is_constructible_from_v = std::is_constructible_v<T, U>;

template <typename T, typename U>
concept is_assignable_from = std::is_assignable_v<T&, U>;

template <typename T, typename U>
concept is_implicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_convertible_v<const U&, T>;

template <typename T, typename U>
concept is_explicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, const U&> && !std::is_convertible_v<const U&, T>;

template <typename T, typename U>
concept is_implicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_explicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_directly_constructible =
    !utils::traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_explicitly_directly_constructible =
    !utils::traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_copy_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

template <typename T, typename U>
concept is_move_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_assignable_v<T&, U&&>;

}  // namespace core::meta::concepts