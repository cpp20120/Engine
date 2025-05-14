#pragma once
#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace core::meta::dynamic_optional {
namespace detail {

/**
 * @brief Trait to check if a type is an optional type.
 * @tparam T The type to check.
 */
template <typename T>
struct is_optional : std::false_type {};

/**
 * @brief Specialization for dynamic_optional type.
 * @tparam T The type contained in the dynamic_optional.
 */
template <typename T>
struct is_optional<dynamic_optional<T>> : std::true_type {};

/**
 * @brief Specialization for std::optional type.
 * @tparam T The type contained in the std::optional.
 */
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

/**
 * @brief Helper variable template for is_optional.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

/**
 * @brief Concept to check if a type is not an optional.
 * @tparam U The type to check.
 */
template <typename U>
concept NotOptional = !is_optional_v<std::decay_t<U>>;

/**
 * @brief Concept to check if a type can be safely converted to another type.
 * @tparam T The target type.
 * @tparam U The source type.
 */
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

/**
 * @brief Storage class for dynamic_optional to handle inline or heap storage.
 * @tparam T The type to store.
 */
template <typename T>
class Storage {
  static constexpr bool use_inline =
      sizeof(T) <= 2 * sizeof(void*) && std::is_nothrow_move_constructible_v<T>;

  using InlineStorage = T;
  using HeapStorage = std::unique_ptr<T>;
  using StorageType =
      std::conditional_t<use_inline, InlineStorage, HeapStorage>;

  std::variant<std::monostate, StorageType> data;

 public:
  Storage() = default;

  ~Storage() { reset(); }

  /**
   * @brief Copy constructor.
   * @param other The Storage object to copy.
   */
  Storage(const Storage& other) {
    if (other.has_value()) {
      emplace(*other);
    }
  }

  Storage(Storage&& other) noexcept = default;

  /**
   * @brief Emplaces a new value in the storage.
   * @tparam Args The argument types.
   * @param args The arguments to forward to the constructor of T.
   */
  template <typename... Args>
  void emplace(Args&&... args) {
    reset();
    if constexpr (use_inline) {
      data.template emplace<StorageType>(std::forward<Args>(args)...);
    } else {
      data.template emplace<StorageType>(
          std::make_unique<T>(std::forward<Args>(args)...));
    }
  }

  /**
   * @brief Resets the storage to an empty state.
   */
  void reset() noexcept { data = std::monostate{}; }

  /**
   * @brief Checks if the storage has a value.
   * @return true if the storage has a value, false otherwise.
   */
  bool has_value() const noexcept { return data.index() != 0; }

  /**
   * @brief Gets the stored value.
   * @return Reference to the stored value.
   */
  T& get() & {
    assert(has_value());
    if constexpr (use_inline) {
      return std::get<StorageType>(data);
    } else {
      return *std::get<StorageType>(data);
    }
  }

  /**
   * @brief Gets the stored value (const version).
   * @return Const reference to the stored value.
   */
  const T& get() const& {
    assert(has_value());
    if constexpr (use_inline) {
      return std::get<StorageType>(data);
    } else {
      return *std::get<StorageType>(data);
    }
  }

  /**
   * @brief Gets the stored value (rvalue version).
   * @return Rvalue reference to the stored value.
   */
  T&& get() && {
    assert(has_value());
    if constexpr (use_inline) {
      return std::move(std::get<StorageType>(data));
    } else {
      return std::move(*std::get<StorageType>(data));
    }
  }
};

}  // namespace detail

/**
 * @brief A dynamic optional type that can store a value or be empty.
 * @tparam T The type of the value to store.
 */
template <typename T>
class dynamic_optional {
 public:
  using value_type = T;

  // Constructors
  dynamic_optional() noexcept = default;
  dynamic_optional(std::nullopt_t) noexcept {}

  /**
   * @brief Copy constructor.
   * @param other The dynamic_optional object to copy.
   */
  dynamic_optional(const dynamic_optional& other) {
    if (other.has_value()) {
      m_storage.emplace(*other);
    }
  }

  dynamic_optional(dynamic_optional&& other) noexcept = default;

  /**
   * @brief Constructs a dynamic_optional from another dynamic_optional of a
   * different type.
   * @tparam U The type of the other dynamic_optional.
   * @param other The dynamic_optional object to copy.
   */
  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U>)
  explicit dynamic_optional(const dynamic_optional<U>& other) {
    if (other.has_value()) {
      m_storage.emplace(*other);
    }
  }

  /**
   * @brief Constructs a dynamic_optional from another dynamic_optional of a
   * different type (move version).
   * @tparam U The type of the other dynamic_optional.
   * @param other The dynamic_optional object to move.
   */
  template <typename U>
    requires(!std::is_same_v<U, T> && detail::SafeConversion<T, U>)
  explicit dynamic_optional(dynamic_optional<U>&& other) {
    if (other.has_value()) {
      m_storage.emplace(std::move(*other));
    }
  }

  /**
   * @brief Constructs a dynamic_optional from a value.
   * @tparam U The type of the value.
   * @param value The value to store.
   */
  template <typename U = T>
    requires(detail::NotOptional<U>)
  explicit(!std::is_convertible_v<U, T>) dynamic_optional(U&& value) {
    m_storage.emplace(std::forward<U>(value));
  }

  // Assignment operators
  dynamic_optional& operator=(std::nullopt_t) noexcept {
    m_storage.reset();
    return *this;
  }

  /**
   * @brief Copy assignment operator.
   * @param other The dynamic_optional object to copy.
   * @return Reference to this object.
   */
  dynamic_optional& operator=(const dynamic_optional& other) {
    if (this != &other) {
      if (other.has_value()) {
        if (has_value()) {
          **this = *other;
        } else {
          m_storage.emplace(*other);
        }
      } else {
        m_storage.reset();
      }
    }
    return *this;
  }

  dynamic_optional& operator=(dynamic_optional&& other) noexcept = default;

  /**
   * @brief Assigns a value to the dynamic_optional.
   * @tparam U The type of the value.
   * @param value The value to store.
   * @return Reference to this object.
   */
  template <typename U>
  dynamic_optional& operator=(U&& value) {
    if (has_value()) {
      **this = std::forward<U>(value);
    } else {
      m_storage.emplace(std::forward<U>(value));
    }
    return *this;
  }

  // Observers
  explicit operator bool() const noexcept { return has_value(); }
  bool has_value() const noexcept { return m_storage.has_value(); }

  /**
   * @brief Gets the stored value (const version).
   * @return Const reference to the stored value.
   * @throws std::bad_optional_access if the dynamic_optional is empty.
   */
  const T& value() const& {
    if (!has_value()) throw std::bad_optional_access{};
    return **this;
  }

  /**
   * @brief Gets the stored value.
   * @return Reference to the stored value.
   * @throws std::bad_optional_access if the dynamic_optional is empty.
   */
  T& value() & {
    if (!has_value()) throw std::bad_optional_access{};
    return **this;
  }

  /**
   * @brief Gets the stored value (rvalue version).
   * @return Rvalue reference to the stored value.
   * @throws std::bad_optional_access if the dynamic_optional is empty.
   */
  T&& value() && {
    if (!has_value()) throw std::bad_optional_access{};
    return std::move(**this);
  }

  /**
   * @brief Gets the stored value (const rvalue version).
   * @return Const rvalue reference to the stored value.
   * @throws std::bad_optional_access if the dynamic_optional is empty.
   */
  const T&& value() const&& {
    if (!has_value()) throw std::bad_optional_access{};
    return std::move(**this);
  }

  /**
   * @brief Gets the stored value or a default value if the dynamic_optional is
   * empty.
   * @tparam U The type of the default value.
   * @param default_value The default value to return if the dynamic_optional is
   * empty.
   * @return The stored value or the default value.
   */
  template <typename U>
  T value_or(U&& default_value) const& {
    static_assert(std::is_convertible_v<U, T>);
    return has_value() ? **this
                       : static_cast<T>(std::forward<U>(default_value));
  }

  /**
   * @brief Gets the stored value or a default value if the dynamic_optional is
   * empty (rvalue version).
   * @tparam U The type of the default value.
   * @param default_value The default value to return if the dynamic_optional is
   * empty.
   * @return The stored value or the default value.
   */
  template <typename U>
  T value_or(U&& default_value) && {
    static_assert(std::is_convertible_v<U, T>);
    return has_value() ? std::move(**this)
                       : static_cast<T>(std::forward<U>(default_value));
  }

  // Modifiers
  void reset() noexcept { m_storage.reset(); }

  /**
   * @brief Emplaces a new value in the dynamic_optional.
   * @tparam Args The argument types.
   * @param args The arguments to forward to the constructor of T.
   * @return Reference to the stored value.
   */
  template <typename... Args>
  T& emplace(Args&&... args) {
    m_storage.emplace(std::forward<Args>(args)...);
    return **this;
  }

  // Accessors
  const T& operator*() const& noexcept { return m_storage.get(); }
  T& operator*() & noexcept { return m_storage.get(); }
  T&& operator*() && noexcept { return std::move(m_storage.get()); }
  const T&& operator*() const&& noexcept { return std::move(m_storage.get()); }

  const T* operator->() const noexcept { return &m_storage.get(); }
  T* operator->() noexcept { return &m_storage.get(); }

  // Swap support
  void swap(dynamic_optional& other) noexcept {
    if (has_value() && other.has_value()) {
      using std::swap;
      swap(**this, *other);
    } else if (has_value()) {
      other.m_storage.emplace(std::move(**this));
      m_storage.reset();
    } else if (other.has_value()) {
      m_storage.emplace(std::move(*other));
      other.m_storage.reset();
    }
  }

 private:
  detail::Storage<T> m_storage;
};

// Comparison operators
template <typename T>
bool operator==(const dynamic_optional<T>& lhs,
                const dynamic_optional<T>& rhs) {
  if (lhs.has_value() != rhs.has_value()) return false;
  return !lhs.has_value() || (*lhs == *rhs);
}

template <typename T>
bool operator!=(const dynamic_optional<T>& lhs,
                const dynamic_optional<T>& rhs) {
  return !(lhs == rhs);
}

// Comparison with nullopt
template <typename T>
bool operator==(const dynamic_optional<T>& opt, std::nullopt_t) noexcept {
  return !opt.has_value();
}

template <typename T>
bool operator==(std::nullopt_t, const dynamic_optional<T>& opt) noexcept {
  return !opt.has_value();
}

// Swap function
template <typename T>
void swap(dynamic_optional<T>& lhs, dynamic_optional<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace core::meta::dynamic_optional
