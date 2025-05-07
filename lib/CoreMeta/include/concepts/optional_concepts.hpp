#pragma

#include <type_traits>
#include "../traits/type_traits.hpp"

namespace core::meta::concepts {

template <typename T, typename U>
concept convertible_to_optional_like =
    core::meta::type_traits::is_convertible_to_optional_like<T, U>::value;

template <typename T, typename U>
concept assingable_from_optional_like =
    core::meta::type_traits::is_assignable_from_optional_like<T, U>::value;

template <typename T>
concept optional_like = core::meta::type_traits::is_optional_like<T>::value;

template <typename T, typename U>
concept assignable_value =
    !core::meta::type_traits::is_optional_like<std::decay_t<U>>::value &&
    std::is_constructible_v<T, U> &&
    core::meta::type_traits::is_assignable_from_optional_like<T, U>::value &&
    !std::conjunction_v<std::is_scalar<T>, std::is_same<T, std::decay_t<U>>>;

template <typename T, typename U>
concept moveable_assign_from =
    !std::is_same_v<U, T> &&
    core::meta::type_traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, U> && std::is_assignable_v<T&, U>;

template <typename T, typename U>
concept copyable_assign_from =
    !std::is_same_v<U, T> &&
    core::meta::type_traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

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

template <typename T, typename U>
concept is_converting_assign =
    is_converting_ctor<T, U> &&
    !std::disjunction_v < std::is_assignable<T&, std::optional<U>&>,
        std::is_assignable<T&, const std::optional<U>&>,
        std::is_assignable<T&, std::optional<U>&&>,
        std::is_assignable<T&, const std::optional<U>&&>
}  // namespace core::meta::concepts