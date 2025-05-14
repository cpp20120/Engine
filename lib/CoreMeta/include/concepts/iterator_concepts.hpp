#pragma once
#include <concepts>
#include <iterator>
#include <type_traits>
#include <utility>

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

// Define the iterator tags for const/mutable distinction
/**
 * @struct const_iterator_tag
 * @brief Tag to indicate a const iterator.
 */
struct const_iterator_tag {};

/**
 * @struct mutable_iterator_tag
 * @brief Tag to indicate a mutable iterator.
 */
struct mutable_iterator_tag {};

/**
 * @concept iterable
 * @brief Concept to check if a type T is iterable.
 * @tparam T The type to check.
 */
template <typename T>
concept iterable = requires(T t) {
  { std::begin(t) } -> std::input_or_output_iterator;
  { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
};

/**
 * @concept iterator
 * @brief Concept to check if a type T is an iterator.
 * @tparam T The type to check.
 */
template <typename T>
concept iterator = requires(T t) {
  { *t } -> std::same_as<std::iter_reference_t<T>>;
  { ++t } -> std::same_as<T&>;
};

/**
 * @concept forward_iterator
 * @brief Concept to check if a type T is a forward iterator.
 * @tparam T The type to check.
 */
template <typename T>
concept forward_iterator = std::forward_iterator<T>;

/**
 * @concept bidirectional_iterator
 * @brief Concept to check if a type T is a bidirectional iterator.
 * @tparam T The type to check.
 */
template <typename T>
concept bidirectional_iterator = std::bidirectional_iterator<T>;

/**
 * @concept random_access_iterator
 * @brief Concept to check if a type T is a random access iterator.
 * @tparam T The type to check.
 */
template <typename T>
concept random_access_iterator = std::random_access_iterator<T>;

/**
 * @class mock_iterator_proxy_reference
 * @brief A mock proxy reference for iterator operations.
 * @tparam T The type of the element.
 * @tparam readable Whether the reference is readable.
 * @tparam writable Whether the reference is writable.
 */
template <typename T, bool readable, bool writable>
class mock_iterator_proxy_reference {
 public:
  /**
   * @brief Assignment operator for const T.
   * @return Reference to this proxy.
   */
  auto operator=(T const&) const -> mock_iterator_proxy_reference&
    requires writable;

  /**
   * @brief Assignment operator for rvalue T.
   * @return Reference to this proxy.
   */
  auto operator=(T&&) const -> mock_iterator_proxy_reference&
    requires writable;

  /**
   * @brief Conversion operator to T.
   */
  operator T() const
    requires readable;

  /**
   * @brief Arrow operator for const T.
   * @return Pointer to const T.
   */
  auto operator->() const -> T const*
    requires readable;

  /**
   * @brief Arrow operator for mutable T.
   * @return Pointer to T.
   */
  auto operator->() const -> T*
    requires readable and writable;
};

/**
 * @struct mock_iterator_value_type_def
 * @brief Defines the value type for a mock iterator.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T, typename IteratorCategory>
struct mock_iterator_value_type_def {};

/**
 * @struct mock_iterator_value_type_def specialization for output iterators.
 * @tparam T The type of the element.
 */
template <typename T>
struct mock_iterator_value_type_def<T, std::output_iterator_tag> {
  using value_type = void;
};

/**
 * @struct mock_iterator_value_type_def specialization for input iterators.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T,
          std::derived_from<std::input_iterator_tag> IteratorCategory>
struct mock_iterator_value_type_def<T, IteratorCategory> {
  using value_type = T;
};

/**
 * @struct mock_iterator_reference_def
 * @brief Defines the reference type for a mock iterator.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 * @tparam writable Whether the iterator is writable.
 */
template <typename T, typename IteratorCategory, bool writable>
struct mock_iterator_reference_def {};

/**
 * @struct mock_iterator_reference_def specialization for output iterators.
 * @tparam T The type of the element.
 */
template <typename T>
struct mock_iterator_reference_def<T, std::output_iterator_tag, true> {
  using reference = void;

 protected:
  using deref_result = mock_iterator_proxy_reference<T, false, true>;
  using arrow_result = void;
};

/**
 * @struct mock_iterator_reference_def specialization for input iterators.
 * @tparam T The type of the element.
 * @tparam writable Whether the iterator is writable.
 */
template <typename T, bool writable>
struct mock_iterator_reference_def<T, std::input_iterator_tag, writable> {
  using reference = mock_iterator_proxy_reference<T, true, writable>;

 protected:
  using deref_result = reference;
  using arrow_result = reference;
};

/**
 * @struct mock_iterator_reference_def specialization for forward iterators.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T,
          std::derived_from<std::forward_iterator_tag> IteratorCategory>
struct mock_iterator_reference_def<T, IteratorCategory, true> {
  using reference = T&;

 protected:
  using deref_result = reference;
  using arrow_result = T*;
};

/**
 * @struct mock_iterator_reference_def specialization for const forward
 * iterators.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T,
          std::derived_from<std::forward_iterator_tag> IteratorCategory>
struct mock_iterator_reference_def<T, IteratorCategory, false> {
  using reference = T const&;

 protected:
  using deref_result = reference;
  using arrow_result = T const*;
};

/**
 * @struct mock_iterator_element_type_def
 * @brief Defines the element type for a mock iterator.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T, typename IteratorCategory>
struct mock_iterator_element_type_def {};

/**
 * @struct mock_iterator_element_type_def specialization for contiguous
 * iterators.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T,
          std::derived_from<std::contiguous_iterator_tag> IteratorCategory>
struct mock_iterator_element_type_def<T, IteratorCategory> {
  using element_type = T;
};

/**
 * @class mock_iterator
 * @brief A mock iterator class for testing purposes.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 * @tparam RWCategory The read/write category.
 */
template <typename T, typename IteratorCategory, typename RWCategory>
class mock_iterator final
    : public mock_iterator_value_type_def<T, IteratorCategory>,
      public mock_iterator_reference_def<
          T, IteratorCategory, std::same_as<RWCategory, mutable_iterator_tag>>,
      public mock_iterator_element_type_def<T, IteratorCategory> {
 public:
  using iterator_category = IteratorCategory;
  using difference_type = std::ptrdiff_t;

  /**
   * @brief Prefix increment operator.
   * @return Reference to this iterator.
   */
  auto operator++() -> mock_iterator& { return *this; }

  /**
   * @brief Postfix increment operator.
   * @return A new iterator.
   */
  auto operator++(int) -> mock_iterator { return {}; }

  /**
   * @brief Dereference operator.
   * @return The dereferenced result.
   */
  auto operator*() const -> typename mock_iterator::deref_result { return {}; }

  /**
   * @brief Arrow operator.
   * @return The arrow result.
   */
  auto operator->() const -> typename mock_iterator::arrow_result
    requires std::derived_from<IteratorCategory, std::input_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Equality operator.
   * @return Whether the iterators are equal.
   */
  auto operator==(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory, std::input_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Prefix decrement operator.
   * @return Reference to this iterator.
   */
  auto operator--() -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::bidirectional_iterator_tag>
  {
    return *this;
  }

  /**
   * @brief Postfix decrement operator.
   * @return A new iterator.
   */
  auto operator--(int) -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::bidirectional_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Addition assignment operator.
   * @return Reference to this iterator.
   */
  auto operator+=(difference_type) -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return *this;
  }

  /**
   * @brief Subtraction assignment operator.
   * @return Reference to this iterator.
   */
  auto operator-=(difference_type) -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return *this;
  }

  /**
   * @brief Subscript operator.
   * @return The dereferenced result.
   */
  auto operator[](difference_type) const -> typename mock_iterator::deref_result
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Addition operator.
   * @return A new iterator.
   */
  auto operator+(difference_type) const -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Friend addition operator.
   * @return A new iterator.
   */
  template <std::convertible_to<difference_type> D>
  friend auto operator+(D const&, mock_iterator const&) -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Subtraction operator.
   * @return A new iterator.
   */
  auto operator-(difference_type const&) const -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Subtraction operator between iterators.
   * @return The difference between iterators.
   */
  auto operator-(mock_iterator const&) const -> difference_type
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Less than operator.
   * @return Whether this iterator is less than the other.
   */
  auto operator<(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Greater than operator.
   * @return Whether this iterator is greater than the other.
   */
  auto operator>(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Less than or equal to operator.
   * @return Whether this iterator is less than or equal to the other.
   */
  auto operator<=(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Greater than or equal to operator.
   * @return Whether this iterator is greater than or equal to the other.
   */
  auto operator>=(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  /**
   * @brief Three-way comparison operator.
   * @return The comparison result.
   */
  auto operator<=>(mock_iterator const&) const
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }
};

/**
 * @typedef mock_const_iterator
 * @brief Typedef for a const mock iterator.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T, typename IteratorCategory>
using mock_const_iterator =
    mock_iterator<T, IteratorCategory, const_iterator_tag>;

/**
 * @typedef mock_mutable_iterator
 * @brief Typedef for a mutable mock iterator.
 * @tparam T The type of the element.
 * @tparam IteratorCategory The iterator category.
 */
template <typename T, typename IteratorCategory>
using mock_mutable_iterator =
    mock_iterator<T, IteratorCategory, mutable_iterator_tag>;

}  // namespace core::meta::concepts
