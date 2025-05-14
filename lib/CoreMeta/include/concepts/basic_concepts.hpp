#pragma once
#include <type_traits>

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept is_number
 * @brief Concept to check if a type T is a number.
 * @tparam T The type to check.
 */
template <typename T>
concept is_number = std::is_arithmetic_v<T>;

/**
 * @concept is_floating_point
 * @brief Concept to check if a type T is a floating-point type.
 * @tparam T The type to check.
 */
template <typename T>
concept is_floating_point = std::is_floating_point_v<T>;

/**
 * @concept is_integral
 * @brief Concept to check if a type T is an integral type.
 * @tparam T The type to check.
 */
template <typename T>
concept is_integral = std::is_integral_v<T>;

/**
 * @concept is_arithmetic
 * @brief Concept to check if a type T is an arithmetic type.
 * @tparam T The type to check.
 */
template <typename T>
concept is_arithmetic = std::is_arithmetic_v<T>;

/**
 * @concept decayed
 * @brief Concept to check if a type T is the same as its decayed type.
 * @tparam T The type to check.
 */
template <typename T>
concept decayed = std::same_as<T, std::decay_t<T>>;

/**
 * @concept aggregate
 * @brief Concept to check if a type T is an aggregate type.
 * @tparam T The type to check.
 */
template <typename T>
concept aggregate = std::is_aggregate_v<T>;

/**
 * @concept trivial
 * @brief Concept to check if a type T is a trivial type.
 * @tparam T The type to check.
 */
template <typename T>
concept trivial = std::is_trivial_v<T>;

/**
 * @concept enum_type
 * @brief Concept to check if a type T is an enum type.
 * @tparam T The type to check.
 */
template <typename T>
concept enum_type = std::is_enum_v<T>;

/**
 * @concept error_code_enum
 * @brief Concept to check if a type T is an error code enum.
 * @tparam T The type to check.
 */
template <typename T>
concept error_code_enum = enum_type<T> and std::is_error_code_enum_v<T>;

/**
 * @concept error_condition_enum
 * @brief Concept to check if a type T is an error condition enum.
 * @tparam T The type to check.
 */
template <typename T>
concept error_condition_enum =
    enum_type<T> and std::is_error_condition_enum_v<T>;

/**
 * @concept try_to_instantiate
 * @brief Concept that always evaluates to true, used for testing instantiation.
 * @tparam ... Types to try to instantiate.
 */
template <typename...>
concept try_to_instantiate = true;
}  // namespace core::meta::concepts
