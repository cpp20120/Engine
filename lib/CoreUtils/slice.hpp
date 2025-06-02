#pragma once

#include <ranges>
#include <span>
#include <type_traits>

namespace core::utils {

template <int left = 0, int right = 0, typename T>
constexpr auto slice(T&& container) {
  static_assert(left >= 0, "Left index must be non-negative");

  using value_type = std::remove_reference_t<T>;
  using span_type =
      std::conditional_t<std::is_const_v<value_type>,
                         std::span<const typename value_type::value_type>,
                         std::span<typename value_type::value_type> >;

  auto begin = std::ranges::begin(std::forward<T>(container)) + left;
  auto end = right > 0 ? begin + (right - left)
                       : std::ranges::end(std::forward<T>(container)) + right;

  if (begin > end || static_cast<std::size_t>(
                         end - std::ranges::begin(std::forward<T>(container))) >
                         std::size(container)) {
    return span_type{};
  }

  return span_type(begin, end);
}

template <typename T>
constexpr auto slice(T&& container, std::size_t left, std::size_t right) {
  using value_type = std::remove_reference_t<T>;
  using span_type =
      std::conditional_t<std::is_const_v<value_type>,
                         std::span<const typename value_type::value_type>,
                         std::span<typename value_type::value_type> >;

  auto begin = std::ranges::begin(std::forward<T>(container)) + left;
  auto end = right > 0 ? begin + (right - left)
                       : std::ranges::end(std::forward<T>(container)) + right;

  if (begin > end || static_cast<std::size_t>(
                         end - std::ranges::begin(std::forward<T>(container))) >
                         std::size(container)) {
    return span_type{};
  }

  return span_type(begin, end);
}

}  // namespace core::utils