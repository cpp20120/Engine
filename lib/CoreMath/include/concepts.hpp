#pragma once

#include <type_traits>
#include <utility>

namespace core::math::concepts {

/**
 * @concept is_number
 * @brief Concept for arithmetic types.
 *
 * This concept checks if a type is arithmetic, which includes integral and
 * floating-point types. It is useful for ensuring that a type can be used in
 * arithmetic operations.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept is_number = std::is_arithmetic_v<T>;

/**
 * @concept is_floating_point
 * @brief Concept for floating-point types.
 *
 * This concept checks if a type is a floating-point type, such as float,
 * double, or long double. It is useful for ensuring that a type can be used in
 * floating-point arithmetic operations.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept is_floating_point = std::is_floating_point_v<T>;

/**
 * @concept is_integral
 * @brief Concept for integral types.
 *
 * This concept checks if a type is an integral type, such as int, long, or
 * short. It is useful for ensuring that a type can be used in integral
 * arithmetic operations.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept is_integral = std::is_integral_v<T>;

}  // namespace core::math::concepts
