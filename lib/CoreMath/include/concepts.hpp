#pragma once

#include <type_traits>
#include <utility>

namespace core::math::concepts {

template <typename T>
concept is_number = std::is_arithmetic_v<T>;

template <typename T>
concept is_floating_point = std::is_floating_point_v<T>;

template <typename T>
concept is_integral = std::is_integral_v<T>;

}  // namespace core::math::concepts
