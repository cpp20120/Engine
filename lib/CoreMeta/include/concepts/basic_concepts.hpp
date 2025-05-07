#pragma once
#include <type_traits>

namespace core::meta::concepts {

template <typename T>
concept is_number = std::is_arithmetic_v<T>;

template <typename T>
concept is_floating_point = std::is_floating_point_v<T>;

template <typename T>
concept is_integral = std::is_integral_v<T>;

template <typename T>
concept is_arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept decayed = std::same_as<T, std::decay_t<T>>;

template <typename T>
concept aggregate = std::is_aggregate_v<T>;

template <typename T>
concept trivial = std::is_trivial_v<T>;

template <typename T>
concept enum_type = std::is_enum_v<T>;

template <typename T>
concept error_code_enum = enum_type<T> and std::is_error_code_enum_v<T>;

template <typename T>
concept error_condition_enum =
    enum_type<T> and std::is_error_condition_enum_v<T>;

template <typename...>
concept try_to_instantiate = true;
}  // namespace core::meta::concepts