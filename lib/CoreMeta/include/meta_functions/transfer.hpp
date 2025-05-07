#pragma once 

#include <type_traits>

namespace core::meta::transfer {

template <typename T, typename U>
struct transfer_reference : std::type_identity<U> {};

template <typename T, typename U>
struct transfer_reference<T&, U> : std::add_lvalue_reference<U> {};

template <typename T, typename U>
struct transfer_reference<T&&, U> : std::add_rvalue_reference<U> {};

template <typename T, typename U>
using transfer_reference_t = typeof_t<transfer_reference<T, U>>;

template <typename T, typename U>
struct _transfer_cv : std::type_identity<U> {};

template <typename T, typename U>
struct _transfer_cv<const T, U>
    : transfer_reference<U, std::add_const_t<std::remove_reference_t<U>>> {};

template <typename T, typename U>
struct _transfer_cv<volatile T, U>
    : transfer_reference<U, std::add_volatile_t<std::remove_reference_t<U>>> {};

template <typename T, typename U>
struct _transfer_cv<const volatile T, U>
    : transfer_reference<U, std::add_cv_t<std::remove_reference_t<U>>> {};

template <typename T, typename U>
struct transfer_cv : _transfer_cv<std::remove_reference_t<T>, U> {};

template <typename T, typename U>
using transfer_cv_t = typeof_t<transfer_cv<T, U>>;

template <typename T, typename U>
struct transfer_cvref : transfer_reference<T, transfer_cv_t<T, U>> {};
}  // namespace core::meta::transfer