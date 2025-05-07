#pragma once

namespace core::meta::concepts {

template <typename T, typename U>
concept equality_comparable_with = requires(const T& A, const U& B) {
  { A == B } -> std::same_as<bool>;
  { B == A } -> std::same_as<bool>;
  { A != B } -> std::same_as<bool>;
  { B != A } -> std::same_as<bool>;
};

template <typename T>
concept equality_comparable = equality_comparable_with<T, T>;

template <typename T, typename U>
concept inequality_comparable = requires(T t, U u) {
  { t != u } -> std::convertible_to<bool>;
};

template <typename T, typename U>
concept less_comparable = requires(T t, U u) {
  { t < u } -> std::convertible_to<bool>;
};

template <typename T, typename U>
concept less_eq_comparable = requires(T t, U u) {
  { t <= u } -> std::convertible_to<bool>;
};

template <typename T, typename U>
concept greater_comparable = requires(T t, U u) {
  { t > u } -> std::convertible_to<bool>;
};

template <typename T, typename U>
concept greater_eq_comparable = requires(T t, U u) {
  { t >= u } -> std::convertible_to<bool>;
};

template <typename T>
concept equality_comparableW = equality_comparable_with<T, T>;
}  // namespace core::meta::concepts