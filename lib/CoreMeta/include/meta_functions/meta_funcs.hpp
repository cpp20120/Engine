#pragma once

#include <any>
#include <functional>
#include <tuple>
#include <utility>
#include <variant>

#include "../concepts/function_concepts.hpp"

namespace core::meta::meta_funcs {
template <bool B, typename T = std::void_t<>>
struct disable_if {};

template <typename T>
struct disable_if<false, T> : std::type_identity<T> {};

template <bool B, typename T = std::void_t<>>
using disable_if_t = typeof_t<disable_if<B, T>>;

template <typename... Args>
inline constexpr auto sizeof_v = sizeof...(Args);

template <typename T>
struct variadic_size;

template <template <typename...> typename T, typename... Args>
struct variadic_size<T<Args...>> {
  static constexpr auto value = sizeof_v<Args...>;
};

template <typename T>
inline constexpr auto variadic_size_v = typev<variadic_size<T>>;

template <typename T, typename = std::void_t<>>
struct sizeof_t : index_t<sizeof(T)> {};

template <auto N, typename T>
struct sizeof_t<index_type<N, T>> : sizeof_t<T> {};

template <typename T>
struct sizeof_t<T, std::void_t<decltype(T::size())>> : index_t<T::size()> {};

template <typename T>
struct sizeof_t<T, std::enable_if_t<is_variadic_type_v<T>>>
    : index_t<variadic_size_v<T>> {};

template <typename T>
struct sizeof_t<T, std::void_t<std::enable_if_t<!is_variadic_type_v<T>>,
                               decltype(T::value + 1)>> : T {};

template <typename T>
inline constexpr auto sizeof_t_v = typev<sizeof_t<T>>;

template <typename T, typename U>
using less_t = bool_<(sizeof_t_v<T> < sizeof_t_v<U>)>;

template <typename T, typename U>
inline constexpr auto less_v = typev<less_t<T, U>>;

template <typename T, typename U>
using less_equal_t = bool_<(sizeof_t_v<T> <= sizeof_t_v<U>)>;

template <typename T, typename U>
inline constexpr auto less_equal_v = typev<less_equal_t<T, U>>;

template <typename T, typename U>
using equal_t = bool_<sizeof_t_v<T> == sizeof_t_v<U>>;

template <typename T, typename U>
inline constexpr auto equal_v = typev<equal_t<T, U>>;

template <typename T, typename U>
using greater_equal_t = bool_<(sizeof_t_v<T> >= sizeof_t_v<U>)>;

template <typename T, typename U>
inline constexpr auto greater_equal_v = typev<greater_equal_t<T, U>>;

template <typename T, typename U>
using greater_t = bool_<(sizeof_t_v<T> > sizeof_t_v<U>)>;

template <typename T, typename U>
inline constexpr auto greater_v = typev<greater_t<T, U>>;

template <typename T, typename U>
inline constexpr auto size_diff = sizeof_t_v<T> - sizeof_t_v<U>;

template <typename T>
struct clear : std::type_identity<T> {};

template <template <typename...> typename T, typename... Args>
struct clear<T<Args...>> : std::type_identity<T<>> {};

template <template <typename, auto...> typename T, typename U, auto... Args>
struct clear<T<U, Args...>> : std::type_identity<T<U>> {};

template <typename T>
using clear_t = typeof_t<clear<T>>;

template <bool B, typename T>
struct clear_if : std::conditional_t<B, clear<T>, std::type_identity<T>> {};

template <bool B, typename T>
using clear_if_t = typeof_t<clear_if<B, T>>;

template <typename T, typename U>
using transfer_cvref_t = typeof_t<transfer_cvref<T, U>>;


template <typename F, typename... Ps>
constexpr decltype(auto) curry(F f, Ps... ps) {
  if constexpr (core::meta::concepts::InvocableWith<F, Ps...>) {
    return std::invoke(f, ps...);
  } else {
    return [f, ps...](auto... qs) -> decltype(auto) {
      return curry(f, ps..., qs...);
    };
  }
}
}  // namespace core::meta::meta_funcs