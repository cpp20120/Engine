#pragma once
#include <cassert>
#include <memory>
#include <optional>

#include "./concepts.hpp"
#include "./type_traits.hpp"

namespace core::meta::dynamic_optional {

template <typename T>
class dynamic_optional;

namespace detail {
template <typename>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<dynamic_optional<T>> : std::true_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

template <typename U>
concept NotOptional = !is_optional_v<std::decay_t<U>>;

template <typename T, class U>
concept SafeConversion =
    !std::is_reference_v<U> &&
    !std::is_constructible_v<T, dynamic_optional<U>&> &&
    !std::is_constructible_v<T, const dynamic_optional<U>&> &&
    !std::is_constructible_v<T, dynamic_optional<U>&&> &&
    !std::is_constructible_v<T, const dynamic_optional<U>&&> &&
    !std::is_convertible_v<dynamic_optional<U>&, T> &&
    !std::is_convertible_v<const dynamic_optional<U>&, T> &&
    !std::is_convertible_v<dynamic_optional<U>&&, T> &&
    !std::is_convertible_v<const dynamic_optional<U>&&, T> &&
    !std::is_assignable_v<T&, dynamic_optional<U>&> &&
    !std::is_assignable_v<T&, const dynamic_optional<U>&> &&
    !std::is_assignable_v<T&, dynamic_optional<U>&&> &&
    !std::is_assignable_v<T&, const dynamic_optional<U>&&>;

template <typename T, class U>
concept ConstructibleAndConvertible =
    std::is_constructible_v<T, U> && std::is_convertible_v<U, T>;

template <typename T, class U>
concept ConstructibleNotConvertible =
    std::is_constructible_v<T, U> && !std::is_convertible_v<U, T>;
}  // namespace detail

template <typename T>
class dynamic_optional {
 public:
  using value_type = T;

  dynamic_optional(std::nullopt_t = std::nullopt) {}

  dynamic_optional(const dynamic_optional& rhs)
      : m_storage(rhs ? std::make_unique<T>(*rhs) : nullptr) {}

  dynamic_optional(dynamic_optional&&) noexcept = default;

  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U> &&
             detail::ConstructibleAndConvertible<T, const U&>)
  explicit dynamic_optional(const dynamic_optional<U>& rhs)
      : m_storage(rhs ? std::make_unique<T>(*rhs) : nullptr) {}

  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U> &&
             detail::ConstructibleNotConvertible<T, const U&>)
  explicit dynamic_optional(const dynamic_optional<U>& rhs)
      : m_storage(rhs ? std::make_unique<T>(*rhs) : nullptr) {}

  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U> &&
             detail::ConstructibleAndConvertible<T, U &&>)
  explicit dynamic_optional(dynamic_optional<U>&& rhs)
      : m_storage(rhs ? std::make_unique<T>(std::move(*rhs)) : nullptr) {}

  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U> &&
             detail::ConstructibleNotConvertible<T, U &&>)
  explicit dynamic_optional(dynamic_optional<U>&& rhs)
      : m_storage(rhs ? std::make_unique<T>(std::move(*rhs)) : nullptr) {}

  template <typename U = T>
    requires(detail::NotOptional<U> &&
             detail::ConstructibleAndConvertible<T, U &&>)
  explicit dynamic_optional(U&& value)
      : m_storage(std::make_unique<T>(std::forward<U>(value))) {}

  template <typename U = T>
    requires(detail::NotOptional<U> &&
             detail::ConstructibleNotConvertible<T, U &&>)
  explicit dynamic_optional(U&& value)
      : m_storage(std::make_unique<T>(std::forward<U>(value))) {}

  dynamic_optional& operator=(std::nullopt_t) {
    m_storage.reset();
    return *this;
  }

  dynamic_optional& operator=(const dynamic_optional& rhs) {
    return *this = dynamic_optional(rhs);
  }

  dynamic_optional& operator=(dynamic_optional&&) noexcept = default;

  template <typename U>
    requires core::meta::copyable_assign_from<T, U>
  dynamic_optional& operator=(const dynamic_optional<U>& rhs) {
    if (!rhs.has_value()) {
      m_storage.reset();
    } else if (has_value()) {
      *m_storage = *rhs;
    } else {
      m_storage = std::make_unique<T>(*rhs);
    }
    return *this;
  }

  template <typename U>
    requires utils::concepts::moveable_assign_from<T, U>
  dynamic_optional& operator=(dynamic_optional<U>&& rhs) {
    if (!rhs.has_value()) {
      m_storage.reset();
    } else if (has_value()) {
      *m_storage = std::move(*rhs);
    } else {
      m_storage = std::make_unique<T>(std::move(*rhs));
    }
    return *this;
  }

  template <typename U>
    requires utils::concepts::assignable_value<T, U>
  dynamic_optional& operator=(U&& value) {
    if (has_value()) {
      *m_storage = std::forward<U>(value);
    } else {
      m_storage = std::make_unique<T>(std::forward<U>(value));
    }
    return *this;
  }

  explicit operator bool() const noexcept { return has_value(); }

  bool has_value() const noexcept { return m_storage != nullptr; }

  const T& value() const {
    if (!has_value()) {
      throw std::bad_optional_access{};
    }
    return **this;
  }

  T& value() {
    if (!has_value()) {
      throw std::bad_optional_access{};
    }
    return **this;
  }

  template <typename U>
  T value_or(U&& u) const& {
    static_assert(std::is_convertible_v<U&&, T>);
    return has_value() ? **this : static_cast<T>(std::forward<U>(u));
  }

  template <typename U>
  T value_or(U&& u) && {
    static_assert(std::is_convertible_v<U&&, T>);
    return has_value() ? std::move(**this) : static_cast<T>(std::forward<U>(u));
  }

  const T& operator*() const noexcept {
    assert(has_value());
    return *m_storage;
  }

  T& operator*() noexcept {
    assert(has_value());
    return *m_storage;
  }

  const T* operator->() const noexcept { return &**this; }

  T* operator->() noexcept { return &**this; }

 private:
  std::unique_ptr<T> m_storage;
};

}  // namespace core::meta::dynamic_optional