#pragma once

#include <type_traits>
#include "../traits/type_traits.hpp"

namespace core::meta::transfer {

/**
 * @brief Template struct to transfer reference qualifiers from one type to
 * another.
 *
 * This struct is used to transfer reference qualifiers (lvalue or rvalue) from
 * type T to type U.
 *
 * @tparam T The source type from which to transfer reference qualifiers.
 * @tparam U The target type to which to transfer reference qualifiers.
 */
template <typename T, typename U>
struct transfer_reference : std::type_identity<U> {};

/**
 * @brief Specialization of transfer_reference for lvalue references.
 *
 * This specialization transfers lvalue reference qualifiers from type T to type
 * U.
 *
 * @tparam T The source type from which to transfer reference qualifiers.
 * @tparam U The target type to which to transfer reference qualifiers.
 */
template <typename T, typename U>
struct transfer_reference<T&, U> : std::add_lvalue_reference<U> {};

/**
 * @brief Specialization of transfer_reference for rvalue references.
 *
 * This specialization transfers rvalue reference qualifiers from type T to type
 * U.
 *
 * @tparam T The source type from which to transfer reference qualifiers.
 * @tparam U The target type to which to transfer reference qualifiers.
 */
template <typename T, typename U>
struct transfer_reference<T&&, U> : std::add_rvalue_reference<U> {};

/**
 * @brief Template alias for transferring reference qualifiers from one type to
 * another.
 *
 * This alias uses the transfer_reference struct to transfer reference
 * qualifiers from type T to type U.
 *
 * @tparam T The source type from which to transfer reference qualifiers.
 * @tparam U The target type to which to transfer reference qualifiers.
 */
template <typename T, typename U>
using transfer_reference_t = typeof_t<transfer_reference<T, U>>;

/**
 * @brief Template struct to transfer cv-qualifiers from one type to another.
 *
 * This struct is used to transfer cv-qualifiers (const and volatile) from type
 * T to type U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
struct _transfer_cv : std::type_identity<U> {};

/**
 * @brief Specialization of _transfer_cv for const types.
 *
 * This specialization transfers const qualifiers from type T to type U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
struct _transfer_cv<const T, U>
    : transfer_reference<U, std::add_const_t<std::remove_reference_t<U>>> {};

/**
 * @brief Specialization of _transfer_cv for volatile types.
 *
 * This specialization transfers volatile qualifiers from type T to type U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
struct _transfer_cv<volatile T, U>
    : transfer_reference<U, std::add_volatile_t<std::remove_reference_t<U>>> {};

/**
 * @brief Specialization of _transfer_cv for const volatile types.
 *
 * This specialization transfers const volatile qualifiers from type T to type
 * U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
struct _transfer_cv<const volatile T, U>
    : transfer_reference<U, std::add_cv_t<std::remove_reference_t<U>>> {};

/**
 * @brief Template struct to transfer cv-qualifiers from one type to another.
 *
 * This struct uses _transfer_cv to transfer cv-qualifiers from type T to type
 * U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
struct transfer_cv : _transfer_cv<std::remove_reference_t<T>, U> {};

/**
 * @brief Template alias for transferring cv-qualifiers from one type to
 * another.
 *
 * This alias uses the transfer_cv struct to transfer cv-qualifiers from type T
 * to type U.
 *
 * @tparam T The source type from which to transfer cv-qualifiers.
 * @tparam U The target type to which to transfer cv-qualifiers.
 */
template <typename T, typename U>
using transfer_cv_t = typeof_t<transfer_cv<T, U>>;

/**
 * @brief Template struct to transfer cv and reference qualifiers from one type
 * to another.
 *
 * This struct uses transfer_reference to transfer cv and reference qualifiers
 * from type T to type U.
 *
 * @tparam T The source type from which to transfer cv and reference qualifiers.
 * @tparam U The target type to which to transfer cv and reference qualifiers.
 */
template <typename T, typename U>
struct transfer_cvref : transfer_reference<T, transfer_cv_t<T, U>> {};

}  // namespace core::meta::transfer