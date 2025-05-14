#pragma once

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept equality_comparable_with
 * @brief Concept to check if types T and U can be compared for equality.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept equality_comparable_with = requires(const T& A, const U& B) {
  { A == B } -> std::same_as<bool>;
  { B == A } -> std::same_as<bool>;
  { A != B } -> std::same_as<bool>;
  { B != A } -> std::same_as<bool>;
};

/**
 * @concept equality_comparable
 * @brief Concept to check if type T can be compared for equality with itself.
 * @tparam T The type to check.
 */
template <typename T>
concept equality_comparable = equality_comparable_with<T, T>;

/**
 * @concept inequality_comparable
 * @brief Concept to check if types T and U can be compared for inequality.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept inequality_comparable = requires(T t, U u) {
  { t != u } -> std::convertible_to<bool>;
};

/**
 * @concept less_comparable
 * @brief Concept to check if types T and U can be compared using the less-than
 * operator.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept less_comparable = requires(T t, U u) {
  { t < u } -> std::convertible_to<bool>;
};

/**
 * @concept less_eq_comparable
 * @brief Concept to check if types T and U can be compared using the
 * less-than-or-equal-to operator.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept less_eq_comparable = requires(T t, U u) {
  { t <= u } -> std::convertible_to<bool>;
};

/**
 * @concept greater_comparable
 * @brief Concept to check if types T and U can be compared using the
 * greater-than operator.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept greater_comparable = requires(T t, U u) {
  { t > u } -> std::convertible_to<bool>;
};

/**
 * @concept greater_eq_comparable
 * @brief Concept to check if types T and U can be compared using the
 * greater-than-or-equal-to operator.
 * @tparam T The first type to compare.
 * @tparam U The second type to compare.
 */
template <typename T, typename U>
concept greater_eq_comparable = requires(T t, U u) {
  { t >= u } -> std::convertible_to<bool>;
};

/**
 * @concept equality_comparableW
 * @brief Concept to check if type T can be compared for equality with itself.
 * @tparam T The type to check.
 */
template <typename T>
concept equality_comparableW = equality_comparable_with<T, T>;
}  // namespace core::meta::concepts
