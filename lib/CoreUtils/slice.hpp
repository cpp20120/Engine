#pragma once

#include <cassert>
#include <concepts>
#include <iostream>
#include <iterator>
#include <list>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace core::utils {

template <typename T>
concept ContiguousContainer =
    std::ranges::contiguous_range<T> && std::ranges::sized_range<T>;

template <typename T>
concept StringLike = std::same_as<std::remove_cvref_t<T>, std::string> ||
                     std::same_as<std::remove_cvref_t<T>, std::string_view>;

template <typename T>
concept FixedSizeArray = std::is_array_v<std::remove_reference_t<T>> &&
                         std::extent_v<std::remove_reference_t<T>> != 0;

template <typename T>
concept GeneralRange =
    std::ranges::range<T> && std::ranges::sized_range<T> &&
    !ContiguousContainer<T> && !StringLike<T> && !FixedSizeArray<T>;

template <typename T>
concept Sliceable =
    ContiguousContainer<T> || StringLike<T> || FixedSizeArray<T>;

template <typename T>
concept SubrangeSliceable = GeneralRange<T>;

template <typename T>
concept HasValueType =
    requires { typename std::remove_cvref_t<T>::value_type; };

template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T,
                      std::void_t<typename std::remove_cvref_t<T>::value_type>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_value_type_v = has_value_type<T>::value;

template <typename T>
struct value_type_helper {
  using U = std::remove_cvref_t<T>;
  using type = std::conditional_t<
      StringLike<U>, char,
      std::conditional_t<FixedSizeArray<U>, std::remove_extent_t<U>,
                         std::conditional_t<has_value_type_v<U>,
                                            typename U::value_type, void>>>;
};

template <typename T, typename = void>
struct safe_value_type {
  static_assert(sizeof(T) == 0, "Type has no value_type");
};

template <typename T>
struct safe_value_type<T, std::enable_if_t<StringLike<T>>> {
  using type = char;
};

template <typename T>
struct safe_value_type<T, std::enable_if_t<FixedSizeArray<T>>> {
  using type = std::remove_extent_t<T>;
};

template <typename T>
struct safe_value_type<T, std::enable_if_t<has_value_type_v<T>>> {
  using type = typename std::remove_cvref_t<T>::value_type;
};

template <typename T>
using value_type_t = typename value_type_helper<T>::type;

template <typename T>
using safe_value_type_t = typename safe_value_type<T>::type;

template <typename T>
using span_type_t =
    std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
                       std::span<const value_type_t<T>>,
                       std::span<value_type_t<T>>>;

namespace detail {

template <typename T, typename Iter>
constexpr bool is_valid_slice(Iter begin, Iter end, T&& container) {
  const auto container_begin = std::ranges::begin(container);
  const auto container_end = std::ranges::end(container);
  const auto container_size = std::ranges::size(container);

  if (begin == end && begin == container_end) return true;

  if (begin == container_end || end == container_begin) return false;

  auto distance_begin_to_end = std::distance(begin, end);
  if (distance_begin_to_end < 0) return false;

  auto distance_container_begin_to_begin =
      std::distance(container_begin, begin);
  if (distance_container_begin_to_begin < 0) return false;

  auto distance_container_begin_to_end = std::distance(container_begin, end);
  if (distance_container_begin_to_end >
      static_cast<std::ptrdiff_t>(container_size)) {
    return false;
  }

  if (static_cast<std::size_t>(distance_begin_to_end) > container_size) {
    return false;
  }

  return true;
}

template <typename T>
constexpr std::ptrdiff_t normalize_index(std::ptrdiff_t index, T&& container) {
  const auto size = std::ranges::size(container);
  if (index < 0) {
    index += size;
    if (index < 0) index = 0;
  } else if (static_cast<std::size_t>(index) > size) {
    index = size;
  }
  return index;
}

template <typename T, typename = void>
struct is_sliceable_impl : std::false_type {};

template <typename T>
struct is_sliceable_impl<T, std::enable_if_t<Sliceable<T>>> : std::true_type {};

template <typename T>
struct is_sliceable_impl<T, std::enable_if_t<SubrangeSliceable<T>>>
    : std::true_type {};

}  // namespace detail

template <int left = 0, int right = 0, Sliceable T>
constexpr auto slice(T&& container) {
  static_assert(
      left >= 0 || right <= 0,
      "Negative left or positive right not allowed in compile-time slicing");

  auto&& c = std::forward<T>(container);
  const auto size = std::ranges::size(c);

  auto begin = std::ranges::begin(c);
  if constexpr (left > 0) {
    begin += left;
  }

  auto end = std::ranges::end(c);
  if constexpr (right < 0) {
    end += right;
  } else if constexpr (right > 0) {
    end = begin + (right - left);
  }

  assert(detail::is_valid_slice(begin, end, c) && "Invalid slice bounds");
  if (!detail::is_valid_slice(begin, end, c)) {
    return span_type_t<T>{};
  }

  return span_type_t<T>(begin, end);
}

template <int left = 0, int right = 0, SubrangeSliceable T>
constexpr auto slice(T&& container) {
  static_assert(
      left >= 0 || right <= 0,
      "Negative left or positive right not allowed in compile-time slicing");

  auto&& c = std::forward<T>(container);
  const auto size = std::ranges::size(c);

  auto begin = std::ranges::begin(c);
  if constexpr (left > 0) {
    begin = std::next(begin, left);
  }

  auto end = std::ranges::end(c);
  if constexpr (right < 0) {
    end = std::next(end, right);
  } else if constexpr (right > 0) {
    end = std::next(begin, right - left);
  }

  // assert(detail::is_valid_slice(begin, end, c) && "Invalid slice bounds");
  if (!detail::is_valid_slice(begin, end, c)) {
    return std::ranges::subrange(std::ranges::begin(c), std::ranges::begin(c));
  }

  return std::ranges::subrange(begin, end);
}

template <typename T,
          typename = std::enable_if_t<detail::is_sliceable_impl<T>::value>>
constexpr auto slice(T&& container, std::ptrdiff_t left, std::ptrdiff_t right) {
  auto&& c = std::forward<T>(container);
  const auto size = std::ranges::size(c);

  left = detail::normalize_index(left, c);
  right = detail::normalize_index(right, c);

  if (left > right) {
    assert(!"Left index must be less than or equal to right index");
    if constexpr (Sliceable<T>) {
      return span_type_t<T>{};
    } else {
      return std::ranges::subrange(std::ranges::begin(c),
                                   std::ranges::begin(c));
    }
  }

  auto begin = std::ranges::begin(c);
  auto end = std::ranges::begin(c);
  if constexpr (Sliceable<T>) {
    begin += left;
    end += right;
  } else {
    begin = std::next(begin, left);
    end = std::next(end, right);
  }

  // assert(detail::is_valid_slice(begin, end, c) && "Invalid slice bounds");

  if constexpr (Sliceable<T>) {
    return span_type_t<T>(begin, end);
  } else {
    return std::ranges::subrange(begin, end);
  }
}

}  // namespace core::utils
