#pragma once

#include "basic_concepts.hpp"
#include "container_concepts.hpp"
#include "iterator_concepts.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept Emplaceable
 * @brief Concept to check if a type C supports emplace operations with given
 * arguments.
 * @tparam C The container type to check.
 * @tparam Iterator The iterator type.
 * @tparam Args The argument types to emplace.
 */
template <typename C, typename Iterator, typename... Args>
concept Emplaceable = requires(C& cont, Iterator const& hint, Args&&... args) {
  {
    cont.emplace_hint(hint, std::forward<Args>(args)...)
  } -> std::same_as<typename C::iterator>;
};

/**
 * @concept Insertable
 * @brief Concept to check if a type C supports insert operations with given
 * value.
 * @tparam C The container type to check.
 * @tparam Iterator The iterator type.
 * @tparam Value The value type to insert.
 */
template <typename C, typename Iterator, typename Value>
concept Insertable = requires(C& cont, Iterator const& hint, Value&& value) {
  {
    cont.insert(hint, std::forward<Value>(value))
  } -> std::same_as<typename C::iterator>;
};

/**
 * @concept Erasable
 * @brief Concept to check if a type C supports erase operations.
 * @tparam C The container type to check.
 * @tparam Iterator The iterator type.
 * @tparam Key The key type to erase.
 */
template <typename C, typename Iterator, typename Key>
concept Erasable = requires(C& cont, Iterator const& pos, Iterator const& first,
                            Iterator const& last, Key const& key) {
  { cont.erase(pos) } -> std::same_as<typename C::iterator>;
  { cont.erase(first, last) } -> std::same_as<typename C::iterator>;
  { cont.erase(key) } -> std::same_as<typename C::size_type>;
};

/**
 * @concept associative_container
 * @brief Concept to check if a type C is an associative container.
 * @tparam C The type to check.
 */
template <typename C>
concept associative_container =
    container<C> and sized_container<C> and clearable_container<C> and
    std::default_initializable<C> and requires(C& cont, C const& const_cont) {
      typename C::key_type;

      requires requires(typename C::key_type const& key) {
        { const_cont.count(key) } -> std::same_as<typename C::size_type>;
        { const_cont.contains(key) } -> std::same_as<bool>;
        { cont.find(key) } -> std::same_as<typename C::iterator>;
        { const_cont.find(key) } -> std::same_as<typename C::const_iterator>;
        {
          cont.equal_range(key)
        }
        -> std::same_as<std::pair<typename C::iterator, typename C::iterator>>;
        {
          const_cont.equal_range(key)
        } -> std::same_as<
            std::pair<typename C::const_iterator, typename C::const_iterator>>;
      };

      requires not std::default_initializable<typename C::value_type> or
                   Emplaceable<C, typename C::const_iterator>;

      requires not std::copyable<typename C::value_type> or
                   requires(
                       typename C::value_type const& value,
                       typename C::const_iterator const& hint,
                       mock_const_iterator<typename C::value_type,
                                           std::input_iterator_tag> const&
                           first,
                       mock_const_iterator<typename C::value_type,
                                           std::input_iterator_tag> const& last,
                       std::initializer_list<typename C::value_type> const&
                           init_list) {
                     C(first, last);
                     C(init_list);
                     cont = init_list;
                     Insertable<C, typename C::const_iterator,
                                typename C::value_type const&>;
                     cont.insert(first, last);
                     cont.insert(init_list);
                     Emplaceable<C, typename C::const_iterator,
                                 typename C::value_type const&>;
                   };

      requires not std::movable<typename C::value_type> or
                   requires(typename C::value_type&& tmp_value,
                            typename C::const_iterator const& hint) {
                     Insertable<C, typename C::const_iterator,
                                typename C::value_type &&>;
                     Emplaceable<C, typename C::const_iterator,
                                 typename C::value_type &&>;
                   };

      Erasable<C, typename C::const_iterator, typename C::key_type>;
    };

/**
 * @concept unique_associative_container
 * @brief Concept to check if a type C is a unique associative container.
 * @tparam C The type to check.
 */
template <typename C>
concept unique_associative_container =
    associative_container<C> and requires(C& cont) {
      requires not std::default_initializable<typename C::value_type> or
                   requires(typename C::const_iterator const& hint) {
                     {
                       cont.emplace()
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                   };

      requires not std::copyable<typename C::value_type> or
                   requires(typename C::value_type const& value) {
                     {
                       cont.insert(value)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(value)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                   };

      requires not std::movable<typename C::value_type> or
                   requires(typename C::value_type&& tmp_value) {
                     {
                       cont.insert(std::move(tmp_value))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::move(tmp_value))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                   };
    };

/**
 * @concept multiple_associative_container
 * @brief Concept to check if a type C is a multiple associative container.
 * @tparam C The type to check.
 */
template <typename C>
concept multiple_associative_container =
    associative_container<C> and requires(C& cont) {
      requires not std::default_initializable<typename C::value_type> or
                   requires(typename C::const_iterator const& hint) {
                     { cont.emplace() } -> std::same_as<typename C::iterator>;
                   };

      requires not std::copyable<typename C::value_type> or
                   requires(typename C::value_type const& value) {
                     {
                       cont.insert(value)
                     } -> std::same_as<typename C::iterator>;
                     {
                       cont.emplace(value)
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::movable<typename C::value_type> or
                   requires(typename C::value_type&& tmp_value) {
                     {
                       cont.insert(std::move(tmp_value))
                     } -> std::same_as<typename C::iterator>;
                     {
                       cont.emplace(std::move(tmp_value))
                     } -> std::same_as<typename C::iterator>;
                   };
    };

/**
 * @concept ordered_associative_container
 * @brief Concept to check if a type C is an ordered associative container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_associative_container =
    associative_container<C> and requires(C& cont, C const& const_cont) {
      typename C::key_compare;
      typename C::value_compare;
      requires std::strict_weak_order<
          typename C::key_compare, typename C::key_type, typename C::key_type>;
      requires std::strict_weak_order<typename C::value_compare,
                                      typename C::value_type,
                                      typename C::value_type>;

      requires not std::totally_ordered<typename C::value_type> or
                   std::totally_ordered<C>;

      { const_cont.key_comp() } -> std::same_as<typename C::key_compare>;
      { const_cont.value_comp() } -> std::same_as<typename C::value_compare>;

      requires requires(typename C::key_compare const& key_comp) {
        C(key_comp);
        requires not std::copyable<typename C::value_type> or
                     requires(
                         mock_const_iterator<typename C::value_type,
                                             std::input_iterator_tag> const&
                             first,
                         mock_const_iterator<typename C::value_type,
                                             std::input_iterator_tag> const&
                             last,
                         std::initializer_list<typename C::value_type> const&
                             init_list) {
                       C(first, last, key_comp);
                       C(init_list, key_comp);
                     };
      };

      requires requires(typename C::key_type const& key) {
        { cont.lower_bound(key) } -> std::same_as<typename C::iterator>;
        {
          const_cont.lower_bound(key)
        } -> std::same_as<typename C::const_iterator>;
        { cont.upper_bound(key) } -> std::same_as<typename C::iterator>;
        {
          const_cont.upper_bound(key)
        } -> std::same_as<typename C::const_iterator>;
      };
    };

/**
 * @concept ordered_unique_associative_container
 * @brief Concept to check if a type C is an ordered unique associative
 * container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_unique_associative_container =
    unique_associative_container<C> and ordered_associative_container<C>;

/**
 * @concept ordered_multiple_associative_container
 * @brief Concept to check if a type C is an ordered multiple associative
 * container.
 * @tparam C The type to check.
 */
template <typename C>
concept ordered_multiple_associative_container =
    multiple_associative_container<C> and ordered_associative_container<C>;

/**
 * @concept unordered_associative_container
 * @brief Concept to check if a type C is an unordered associative container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_associative_container =
    associative_container<C> and requires(C& cont, C const& const_cont) {
      typename C::hasher;
      requires hash_function<typename C::hasher, typename C::key_type>;

      typename C::key_equal;
      requires std::equivalence_relation<
          typename C::key_equal, typename C::key_type, typename C::key_type>;

      { const_cont.hash_function() } -> std::same_as<typename C::hasher>;
      { const_cont.key_eq() } -> std::same_as<typename C::key_equal>;

      requires requires(float const& ml, std::size_t n) {
        { const_cont.load_factor() } -> std::same_as<float>;
        { const_cont.max_load_factor() } -> std::same_as<float>;
        cont.max_load_factor(ml);
        cont.rehash(n);
        cont.reserve(n);
      };

      { const_cont.bucket_count() } -> std::same_as<std::size_t>;

      requires requires(std::size_t const& bucket_count,
                        typename C::hasher const& hash,
                        typename C::key_equal const& equal) {
        C(bucket_count);
        C(bucket_count, hash);
        C(bucket_count, hash, equal);

        requires not std::copyable<typename C::value_type> or
                     requires(
                         mock_const_iterator<typename C::value_type,
                                             std::input_iterator_tag> const&
                             first,
                         mock_const_iterator<typename C::value_type,
                                             std::input_iterator_tag> const&
                             last,
                         std::initializer_list<typename C::value_type> const&
                             init_list) {
                       C(first, last, bucket_count);
                       C(first, last, bucket_count, hash);
                       C(first, last, bucket_count, hash, equal);
                       C(init_list, bucket_count);
                       C(init_list, bucket_count, hash);
                       C(init_list, bucket_count, hash, equal);
                     };
      };
    };

/**
 * @concept unordered_unique_associative_container
 * @brief Concept to check if a type C is an unordered unique associative
 * container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_unique_associative_container =
    unique_associative_container<C> and unordered_associative_container<C>;

/**
 * @concept unordered_multiple_associative_container
 * @brief Concept to check if a type C is an unordered multiple associative
 * container.
 * @tparam C The type to check.
 */
template <typename C>
concept unordered_multiple_associative_container =
    multiple_associative_container<C> and unordered_associative_container<C>;

}  // namespace core::meta::concepts
