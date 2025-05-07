#pragma once
#include <iterator>

namespace core::meta::concepts {
template <typename T>
concept iterable = requires(T t) {
  { std::begin(t) } -> std::input_or_output_iterator;
  { std::end(t) } -> std::sentinel_for<decltype(std::begin(t))>;
};

template <typename T>
concept iterator = requires(T t) {
  { *t } -> std::same_as<std::iter_reference_t<T>>;
  { ++t } -> std::same_as<T&>;
};

template <typename T>
concept forward_iterator = std::forward_iterator<T>;

template <typename T>
concept bidirectional_iterator = std::bidirectional_iterator<T>;

template <typename T>
concept random_access_iterator = std::random_access_iterator<T>;


template <typename T, bool readable, bool writable>
class mock_iterator_proxy_reference {
 public:
  auto operator=(T const&) const -> mock_iterator_proxy_reference&
    requires writable;

  auto operator=(T&&) const -> mock_iterator_proxy_reference&
    requires writable;

  operator T() const
    requires readable;

  auto operator->() const -> T const*
    requires readable;

  auto operator->() const -> T*
    requires readable and writable;
};

template <typename T, typename IteratorCategory>
struct mock_iterator_value_type_def {};

template <typename T>
struct mock_iterator_value_type_def<T, std::output_iterator_tag> {
  using value_type = void;
};

template <typename T,
          std::derived_from<std::input_iterator_tag> IteratorCategory>
struct mock_iterator_value_type_def<T, IteratorCategory> {
  using value_type = T;
};

template <typename T, typename IteratorCategory, bool writable>
struct mock_iterator_reference_def {};

template <typename T>
struct mock_iterator_reference_def<T, std::output_iterator_tag, true> {
  using reference = void;

 protected:
  using deref_result = mock_iterator_proxy_reference<T, false, true>;
  using arrow_result = void;
};

template <typename T, bool writable>
struct mock_iterator_reference_def<T, std::input_iterator_tag, writable> {
  using reference = mock_iterator_proxy_reference<T, true, writable>;

 protected:
  using deref_result = reference;
  using arrow_result = reference;
};

template <typename T,
          std::derived_from<std::forward_iterator_tag> IteratorCategory>
struct mock_iterator_reference_def<T, IteratorCategory, true> {
  using reference = T&;

 protected:
  using deref_result = reference;
  using arrow_result = T*;
};

template <typename T,
          std::derived_from<std::forward_iterator_tag> IteratorCategory>
struct mock_iterator_reference_def<T, IteratorCategory, false> {
  using reference = T const&;

 protected:
  using deref_result = reference;
  using arrow_result = T const*;
};

template <typename T, typename IteratorCategory>
struct mock_iterator_element_type_def {};

template <typename T,
          std::derived_from<std::contiguous_iterator_tag> IteratorCategory>
struct mock_iterator_element_type_def<T, IteratorCategory> {
 using element_type = T;
};

template <typename T, typename IteratorCategory, typename RWCategory>
class mock_iterator final
    : public mock_iterator_value_type_def<T, IteratorCategory>,
      public mock_iterator_reference_def<
          T, IteratorCategory, std::same_as<RWCategory, mutable_iterator_tag>>,
      public mock_iterator_element_type_def<T, IteratorCategory> {
 public:
  using iterator_category = IteratorCategory;
  using difference_type = std::ptrdiff_t;

  auto operator++() -> mock_iterator& { return {}; }

  auto operator++(int) -> mock_iterator { return {}; }

  auto operator*() const -> typename mock_iterator::deref_result { return {}; }

  auto operator->() const -> typename mock_iterator::arrow_result
    requires std::derived_from<IteratorCategory, std::input_iterator_tag>
  {
    return {};
  }

  auto operator==(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory, std::input_iterator_tag>
  {
    return {};
  }

  auto operator--() -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::bidirectional_iterator_tag>
  {
    return {};
  }

  auto operator--(int) -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::bidirectional_iterator_tag>
  {
    return {};
  }

  auto operator+=(difference_type) -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator-=(difference_type) -> mock_iterator&
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator[](difference_type) const -> typename mock_iterator::deref_result
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator+(difference_type) const -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  template <std::convertible_to<difference_type> D>
  friend auto operator+(D const&, mock_iterator const&) -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator-(difference_type const&) const -> mock_iterator
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator-(mock_iterator const&) const -> difference_type
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator<(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator>(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator<=(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator>=(mock_iterator const&) const -> bool
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }

  auto operator<=>(mock_iterator const&) const
    requires std::derived_from<IteratorCategory,
                               std::random_access_iterator_tag>
  {
    return {};
  }
};

template <typename T, typename IteratorCategory>
using mock_const_iterator =
    mock_iterator<T, IteratorCategory, const_iterator_tag>;

template <typename T, typename IteratorCategory>
using mock_mutable_iterator =
    mock_iterator<T, IteratorCategory, mutable_iterator_tag>;

}  // namespace core::meta::concepts