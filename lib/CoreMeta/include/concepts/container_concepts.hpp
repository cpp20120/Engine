#pragma once

#include "basic_concepts.hpp"
#include "iterator_concepts.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept container
 * @brief Concept to check if a type C is a container.
 * @tparam C The type to check.
 */
template <typename C>
concept container = requires(C& cont, C const& const_cont) {
  typename C::value_type;
  requires decayed<typename C::value_type>;

  typename C::reference;
  typename C::const_reference;

  requires std::same_as<typename C::reference, typename C::value_type&>;
  requires std::same_as<typename C::const_reference,
                        typename C::value_type const&>;

  typename C::iterator;
  typename C::const_iterator;
  requires std::forward_iterator<typename C::iterator>;
  requires std::forward_iterator<typename C::const_iterator>;
  requires std::convertible_to<typename C::iterator,
                               typename C::const_iterator>;
  requires std::same_as<std::iter_value_t<typename C::iterator>,
                        typename C::value_type>;
  requires std::same_as<std::iter_value_t<typename C::const_iterator>,
                        typename C::value_type>;
  requires std::same_as<std::iter_reference_t<typename C::iterator>,
                        typename C::reference> or
               std::same_as<std::iter_reference_t<typename C::iterator>,
                            typename C::const_reference>;
  requires std::same_as<std::iter_reference_t<typename C::const_iterator>,
                        typename C::const_reference>;

  typename C::difference_type;
  typename C::size_type;
  requires std::signed_integral<typename C::difference_type>;
  requires std::unsigned_integral<typename C::size_type>;

  requires std::in_range<typename C::size_type>(
      std::numeric_limits<typename C::difference_type>::max());

  requires std::same_as<
      typename C::difference_type,
      typename std::iterator_traits<typename C::iterator>::difference_type>;
  requires std::same_as<typename C::difference_type,
                        typename std::iterator_traits<
                            typename C::const_iterator>::difference_type>;

  requires not std::equality_comparable<typename C::value_type> or
               std::equality_comparable<C>;

  requires not std::movable<typename C::value_type> or std::movable<C>;
  requires not std::copyable<typename C::value_type> or std::copyable<C>;
  requires not std::semiregular<typename C::value_type> or std::semiregular<C>;
  requires not std::regular<typename C::value_type> or std::regular<C>;

  // Iterators
  { cont.begin() } -> std::same_as<typename C::iterator>;
  { cont.end() } -> std::same_as<typename C::iterator>;
  { const_cont.begin() } -> std::same_as<typename C::const_iterator>;
  { const_cont.end() } -> std::same_as<typename C::const_iterator>;
  { cont.cbegin() } -> std::same_as<typename C::const_iterator>;
  { cont.cend() } -> std::same_as<typename C::const_iterator>;

  // Capacity
  { const_cont.max_size() } -> std::same_as<typename C::size_type>;
  { const_cont.empty() } -> std::convertible_to<bool>;
};

/**
 * @concept mutable_container
 * @brief Concept to check if a type C is a mutable container.
 * @tparam C The type to check.
 */
template <typename C>
concept mutable_container =
    container<C> and std::same_as<std::iter_reference_t<typename C::iterator>,
                                  typename C::reference>;

/**
 * @concept sized_container
 * @brief Concept to check if a type C is a sized container.
 * @tparam C The type to check.
 */
template <typename C>
concept sized_container = container<C> and requires(C const& const_cont) {
  { const_cont.size() } -> std::same_as<typename C::size_type>;
};

/**
 * @concept clearable_container
 * @brief Concept to check if a type C is a clearable container.
 * @tparam C The type to check.
 */
template <typename C>
concept clearable_container =
    container<C> and requires(C& cont) { cont.clear(); };

/**
 * @concept reversible_container
 * @brief Concept to check if a type C is a reversible container.
 * @tparam C The type to check.
 */
template <typename C>
concept reversible_container =
    container<C> and requires(C& cont, C const& const_cont) {
      requires std::bidirectional_iterator<typename C::iterator>;
      requires std::bidirectional_iterator<typename C::const_iterator>;

      typename C::reverse_iterator;
      typename C::const_reverse_iterator;
      requires std::bidirectional_iterator<typename C::reverse_iterator>;
      requires std::bidirectional_iterator<typename C::const_reverse_iterator>;
      requires std::convertible_to<typename C::reverse_iterator,
                                   typename C::const_reverse_iterator>;
      requires std::same_as<typename C::difference_type,
                            typename std::iterator_traits<
                                typename C::reverse_iterator>::difference_type>;
      requires std::same_as<
          typename C::difference_type,
          typename std::iterator_traits<
              typename C::const_reverse_iterator>::difference_type>;

      { cont.rbegin() } -> std::same_as<typename C::reverse_iterator>;
      { cont.rend() } -> std::same_as<typename C::reverse_iterator>;
      {
        const_cont.rbegin()
      } -> std::same_as<typename C::const_reverse_iterator>;
      { const_cont.rend() } -> std::same_as<typename C::const_reverse_iterator>;
      { cont.crbegin() } -> std::same_as<typename C::const_reverse_iterator>;
      { cont.crend() } -> std::same_as<typename C::const_reverse_iterator>;
    };

}  // namespace core::meta::concepts
