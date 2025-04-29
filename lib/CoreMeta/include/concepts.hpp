#pragma once
#include <functional>
#include <initializer_list>
#include <iterator>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

#include "./type_traits.hpp"

namespace core::meta::concepts {
// ==================== Concepts ====================
template <typename T>
concept is_number = std::is_arithmetic_v<T>;

template <typename T>
concept is_floating_point = std::is_floating_point_v<T>;

template <typename T>
concept is_integral = std::is_integral_v<T>;

template <typename T>
concept is_arithmetic = std::is_arithmetic_v<T>;

template <typename Type>
concept is_awaitable = requires(Type t, std::coroutine_handle<> h) {
  { t.await_read() } -> std::convertible_to<bool>;
  requires std::same_as<decltype(t.await_suspend(h)), void> ||
               std::same_as<decltype(t.await_suspend(h)), bool> ||
               utils::traits::is_coroutine_handle<
                   std::remove_cvref_t<decltype(t.await_suspend(h))>>::value;
  t.await_resume();
};

template <typename T, typename U>
concept equality_comparable_with = requires(const T& A, const U& B) {
  { A == B } -> std::same_as<bool>;
  { B == A } -> std::same_as<bool>;
  { A != B } -> std::same_as<bool>;
  { B != A } -> std::same_as<bool>;
};

template <typename T, typename... Options>
concept any_of = (std::same_as<T, Options> || ...);

template <typename T>
concept equality_comparableW = equality_comparable_with<T, T>;

template <typename T>
concept has_update = requires(T t, float deltaTime) {
  { t.update(deltaTime) } -> std::same_as<void>;
};

template <typename T>
concept drawable = requires(T t, void* context) {
  { t.draw(context) } -> std::same_as<void>;
};

template <typename T>
concept serializable = requires(T t) {
  { t.serialize() } -> std::same_as<std::string>;  // serializing to a string
  { t.deserialize(std::declval<std::string>()) } -> std::same_as<void>;
};

template <typename T>
concept input_handler = requires(T t) {
  { t.processInput() } -> std::same_as<void>;
};

template <typename T, typename U>
concept is_converting_ctor_v = utils::traits::is_converting_ctor<T, U>::value;

template <typename T, typename U>
concept is_converting_assign_v =
    utils::traits::is_converting_assign<T, U>::value;

template <typename T, typename U>
concept is_constructible_from_v = std::is_constructible<T, U>::value;

template <typename T, typename U>
concept is_assignable_from = std::is_assignable<T&, U>::value;

template <typename T, typename U>
concept is_not_same_as = !std::is_same<T, U>::value;

template <typename T, typename U>
concept is_not_optional = !utils::traits::is_optional<std::decay_t<U>>::value;

template <typename...>
concept try_to_instantiate = true;

template <template <typename...> class Expression, typename... Ts>
concept is_detected_impl = requires { typename Expression<Ts...>; };

template <template <typename...> class Expression, typename... Ts>
concept is_detected = is_detected_impl<Expression, Ts...>;

template <typename T, typename U>
concept assignable = requires(T&& t, U&& u) {
  { std::declval<T&>() = std::declval<U&>() } -> std::same_as<T&>;
};

template <typename T, typename... Pack>
concept is_type_in_pack = (std::is_same_v<T, Pack> || ...);

template <typename... Pack>
concept is_pack_uniform =
    (std::is_same_v<utils::traits::peel_first_t<Pack...>, Pack> && ...);

template <typename T, typename... Pack>
concept is_pack_only = is_pack_uniform<Pack...> &&
                       std::is_same_v<T, utils::traits::peel_first_t<Pack...>>;

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

template <typename T>
concept tuple =
    requires { typename std::tuple_size<std::remove_cvref_t<T>>::type; };

template <typename T>
concept pair = requires {
  typename std::tuple_element<0, std::remove_cvref_t<T>>::type;
  typename std::tuple_element<1, std::remove_cvref_t<T>>::type;
};

template <typename T>
concept smart_pointer = requires(T t) {
  { t.get() } -> std::same_as<typename T::element_type*>;
};

template <typename T>
concept printable = requires(std::ostream& os, T t) {
  { os << t } -> std::same_as<std::ostream&>;
};

template <typename T, typename U>
concept equality_comparable = requires(T t, U u) {
  { t == u } -> std::convertible_to<bool>;
};

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

template <std::size_t i>
struct convertible_to_anything {
  template <typename T>
  operator T(){};
};

template <typename T, std::size_t I>
concept is_constructable_with = requires {
  []<std::size_t... is>(std::index_sequence<is...> i_s)
      -> decltype(T{convertible_to_anything<is>{}...}) {
    return {};
  }(std::make_index_sequence<I>{});
};

template <typename T, std::size_t n>
concept aggregate_of = std::is_aggregate_v<T> && is_constructable_with<T, n> &&
                       !is_constructable_with<T, n + 1>;

constexpr std::size_t maxAggregateMembers = 12;

template <typename T>
constexpr auto number_of_aggregate_members =
    []<std::size_t... indexes>(std::index_sequence<indexes...> i_s) {
      return ((aggregate_of<T, indexes> * indexes) + ... + 0);
    }(std::make_index_sequence<maxAggregateMembers>{});

template <typename F, typename... Args>
concept InvocableWith =
    requires(F f, Args... args) { std::invoke(f, args...); };

template <typename F, typename... Ps>
constexpr decltype(auto) curry(F f, Ps... ps) {
  if constexpr (InvocableWith<F, Ps...>) {
    return std::invoke(f, ps...);
  } else {
    return [f, ps...](auto... qs) -> decltype(auto) {
      return curry(f, ps..., qs...);
    };
  }
}

template <typename F>
concept Curryable = requires(F f) {
  {
    std::invoke(f,
                std::declval<std::decay_t<decltype(std::placeholders::_1)>>())
  } -> std::same_as<std::invoke_result_t<F, decltype(std::placeholders::_1)>>;
};

template <typename T>
concept decayed = std::same_as<T, std::decay_t<T>>;

template <typename T>
concept aggregate = std::is_aggregate_v<T>;

template <typename T>
concept trivial = std::is_trivial_v<T>;

template <typename T>
concept enum_type = std::is_enum_v<T>;

template <typename T>
concept error_code_enum = enum_type<T> and std::is_error_code_enum_v<T>;

template <typename T>
concept error_condition_enum =
    enum_type<T> and std::is_error_condition_enum_v<T>;

template <typename Fn, typename... Signatures>
concept invocable_as = requires(Signatures&... signatures) {
  ([]<typename Ret, typename... Args>(auto (&)(Args...)->Ret)
     requires std::is_invocable_r_v<Ret, Fn, Args...>
   {}(signatures),
   ...);
};

template <typename Fn, typename... Signatures>
concept callable_as = requires(Signatures&... signatures) {
  ([]<typename Ret, typename... Args>(auto (&)(Args...)->Ret)
     requires utils::traits::is_callable_r_v<Ret, Fn, Args...>
   {}(signatures),
   ...);
};

template <typename Fn, typename KeyType>
concept hash_function = callable_as<Fn const, auto(KeyType&)->std::size_t,
                                    auto(KeyType const&)->std::size_t>;

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

template <typename C>
concept mutable_container =
    container<C> and std::same_as<std::iter_reference_t<typename C::iterator>,
                                  typename C::reference>;

template <typename C>
concept sized_container = container<C> and requires(C const& const_cont) {
  { const_cont.size() } -> std::same_as<typename C::size_type>;
};

template <typename C>
concept clearable_container =
    container<C> and requires(C& cont) { cont.clear(); };

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

template <typename C, typename ValueType>
concept container_of =
    container<C> and std::same_as<ValueType, typename C::value_type>;

template <typename C, typename ValueType>
concept mutable_container_of =
    container_of<C, ValueType> and mutable_container<C>;

template <typename C, typename ValueType>
concept sized_container_of = container_of<C, ValueType> and sized_container<C>;

template <typename C, typename ValueType>
concept clearable_container_of =
    container_of<C, ValueType> and clearable_container<C>;

template <typename C, typename ValueType>
concept reversible_container_of =
    container_of<C, ValueType> and reversible_container<C>;

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
  // using element_type = T;
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
                   requires(typename C::const_iterator const& hint) {
                     {
                       cont.emplace_hint(hint)
                     } -> std::same_as<typename C::iterator>;
                   };

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

                     {
                       cont.insert(hint, value)
                     } -> std::same_as<typename C::iterator>;

                     cont.insert(first, last);
                     cont.insert(init_list);

                     {
                       cont.emplace_hint(hint, value)
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::movable<typename C::value_type> or
                   requires(typename C::value_type&& tmp_value,
                            typename C::const_iterator const& hint) {
                     {
                       cont.insert(hint, std::move(tmp_value))
                     } -> std::same_as<typename C::iterator>;

                     {
                       cont.emplace_hint(hint, std::move(tmp_value))
                     } -> std::same_as<typename C::iterator>;
                   };

      requires requires(typename C::const_iterator const& pos,
                        typename C::const_iterator const& first,
                        typename C::const_iterator const& last,
                        typename C::key_type const& key) {
        { cont.erase(pos) } -> std::same_as<typename C::iterator>;
        { cont.erase(first, last) } -> std::same_as<typename C::iterator>;
        { cont.erase(key) } -> std::same_as<typename C::size_type>;
      };
    };

template <typename C>
concept unique_associative_container =
    associative_container<C> and requires(C& cont, C const& const_cont) {
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

template <typename C>
concept multiple_associative_container =
    associative_container<C> and requires(C& cont, C const& const_cont) {
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

template <typename C>
concept ordered_unique_associative_container =
    unique_associative_container<C> and ordered_associative_container<C>;

template <typename C>
concept ordered_multiple_associative_container =
    multiple_associative_container<C> and ordered_associative_container<C>;

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

template <typename C>
concept unordered_unique_associative_container =
    unique_associative_container<C> and unordered_associative_container<C>;

template <typename C>
concept unordered_multiple_associative_container =
    multiple_associative_container<C> and unordered_associative_container<C>;

template <typename C>
concept map_container = associative_container<C> and requires(
                                                         C& cont,
                                                         C const& const_cont) {
  typename C::mapped_type;

  requires requires(typename C::const_iterator const& hint) {
    requires not std::copyable<typename C::key_type> or
                 requires(typename C::key_type const& key) {
                   requires not std::default_initializable<
                                typename C::mapped_type> or
                                requires {
                                  {
                                    cont.emplace_hint(hint, key)
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace_hint(
                                        hint, std::piecewise_construct,
                                        std::forward_as_tuple(key),
                                        std::forward_as_tuple())
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::copyable<typename C::mapped_type> or
                                requires(typename C::mapped_type const& obj) {
                                  {
                                    cont.emplace_hint(hint, key, obj)
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace_hint(
                                        hint, std::piecewise_construct,
                                        std::forward_as_tuple(key),
                                        std::forward_as_tuple(obj))
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::movable<typename C::mapped_type> or
                                requires(typename C::mapped_type&& obj) {
                                  {
                                    cont.emplace_hint(hint, key, std::move(obj))
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace_hint(
                                        hint, std::piecewise_construct,
                                        std::forward_as_tuple(key),
                                        std::forward_as_tuple(std::move(obj)))
                                  } -> std::same_as<typename C::iterator>;
                                };
                 };

    requires not std::movable<typename C::key_type> or requires(
                                                           typename C::
                                                               key_type&& key) {
      requires not std::default_initializable<typename C::mapped_type> or
                   requires {
                     {
                       cont.emplace_hint(hint, std::move(key))
                     } -> std::same_as<typename C::iterator>;
                     {
                       cont.emplace_hint(hint, std::piecewise_construct,
                                         std::forward_as_tuple(std::move(key)),
                                         std::forward_as_tuple())
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::copyable<typename C::mapped_type> or
                   requires(typename C::mapped_type const& obj) {
                     {
                       cont.emplace_hint(hint, std::move(key), obj)
                     } -> std::same_as<typename C::iterator>;
                     {
                       cont.emplace_hint(hint, std::piecewise_construct,
                                         std::forward_as_tuple(std::move(key)),
                                         std::forward_as_tuple(obj))
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::movable<typename C::mapped_type> or
                   requires(typename C::mapped_type&& obj) {
                     {
                       cont.emplace_hint(hint, std::move(key), std::move(obj))
                     } -> std::same_as<typename C::iterator>;
                     {
                       cont.emplace_hint(hint, std::piecewise_construct,
                                         std::forward_as_tuple(std::move(key)),
                                         std::forward_as_tuple(std::move(obj)))
                     } -> std::same_as<typename C::iterator>;
                   };
    };
  };
};

template <typename C>
concept unique_map_container = map_container<C> and requires(
                                                        C& cont,
                                                        C const& const_cont) {
  requires requires(typename C::key_type const& key,
                    typename C::key_type const& tmp_key) {
    requires not std::default_initializable<typename C::mapped_type> or
                 requires {
                   requires not std::copyable<typename C::key_type> or
                                requires {
                                  {
                                    cont[key]
                                  } -> std::same_as<typename C::mapped_type&>;
                                };

                   requires not std::movable<typename C::key_type> or requires {
                     {
                       cont[std::move(tmp_key)]
                     } -> std::same_as<typename C::mapped_type&>;
                   };
                 };

    { cont.at(key) } -> std::same_as<typename C::mapped_type&>;
    { const_cont.at(key) } -> std::same_as<typename C::mapped_type const&>;
  };

  requires requires(typename C::const_iterator const& hint) {
    requires not std::copyable<typename C::key_type> or requires(
                                                            typename C::
                                                                key_type const&
                                                                    key) {
      requires not std::default_initializable<typename C::mapped_type> or
                   requires {
                     {
                       cont.emplace(key)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(key),
                                    std::forward_as_tuple())
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(key)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, key)
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::copyable<typename C::mapped_type> or
                   requires(typename C::mapped_type const& obj) {
                     {
                       cont.insert_or_assign(key, obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.insert_or_assign(hint, key, obj)
                     } -> std::same_as<typename C::iterator>;

                     {
                       cont.emplace(key, obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(key),
                                    std::forward_as_tuple(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(key, obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, key, obj)
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::movable<typename C::mapped_type> or
                   requires(typename C::mapped_type&& obj) {
                     {
                       cont.insert_or_assign(key, std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.insert_or_assign(hint, key, std::move(obj))
                     } -> std::same_as<typename C::iterator>;

                     {
                       cont.emplace(key, std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(key),
                                    std::forward_as_tuple(std::move(obj)))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(key, std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, key, std::move(obj))
                     } -> std::same_as<typename C::iterator>;
                   };
    };

    requires not std::movable<typename C::key_type> or requires(
                                                           typename C::
                                                               key_type&& key) {
      requires not std::default_initializable<typename C::mapped_type> or
                   requires {
                     {
                       cont.emplace(std::move(key))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(std::move(key)),
                                    std::forward_as_tuple())
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(std::move(key))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, std::move(key))
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::copyable<typename C::mapped_type> or
                   requires(typename C::mapped_type const& obj) {
                     {
                       cont.insert_or_assign(std::move(key), obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.insert_or_assign(hint, std::move(key), obj)
                     } -> std::same_as<typename C::iterator>;

                     {
                       cont.emplace(std::move(key), obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(std::move(key)),
                                    std::forward_as_tuple(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(std::move(key), obj)
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, std::move(key), obj)
                     } -> std::same_as<typename C::iterator>;
                   };

      requires not std::movable<typename C::mapped_type> or
                   requires(typename C::mapped_type&& obj) {
                     {
                       cont.insert_or_assign(std::move(key), std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.insert_or_assign(hint, std::move(key),
                                             std::move(obj))
                     } -> std::same_as<typename C::iterator>;

                     {
                       cont.emplace(std::move(key), std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(std::move(key)),
                                    std::forward_as_tuple(std::move(obj)))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(std::move(key), std::move(obj))
                     } -> std::same_as<std::pair<typename C::iterator, bool>>;
                     {
                       cont.try_emplace(hint, std::move(key), std::move(obj))
                     } -> std::same_as<typename C::iterator>;
                   };
    };
  };
};

template <typename C>
concept multiple_map_container = map_container<C> and requires(
                                                          C& cont,
                                                          C const& const_cont) {
  requires requires(typename C::const_iterator const& hint) {
    requires not std::copyable<typename C::key_type> or
                 requires(typename C::key_type const& key) {
                   requires not std::default_initializable<
                                typename C::mapped_type> or
                                requires {
                                  {
                                    cont.emplace(key)
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(std::piecewise_construct,
                                                 std::forward_as_tuple(key),
                                                 std::forward_as_tuple())
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::copyable<typename C::mapped_type> or
                                requires(typename C::mapped_type const& obj) {
                                  {
                                    cont.emplace(key, obj)
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(std::piecewise_construct,
                                                 std::forward_as_tuple(key),
                                                 std::forward_as_tuple(obj))
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::movable<typename C::mapped_type> or
                                requires(typename C::mapped_type&& obj) {
                                  {
                                    cont.emplace(key, std::move(obj))
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(
                                        std::piecewise_construct,
                                        std::forward_as_tuple(key),
                                        std::forward_as_tuple(std::move(obj)))
                                  } -> std::same_as<typename C::iterator>;
                                };
                 };

    requires not std::movable<typename C::key_type> or
                 requires(typename C::key_type&& key) {
                   requires not std::default_initializable<
                                typename C::mapped_type> or
                                requires {
                                  {
                                    cont.emplace(std::move(key))
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(
                                        std::piecewise_construct,
                                        std::forward_as_tuple(std::move(key)),
                                        std::forward_as_tuple())
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::copyable<typename C::mapped_type> or
                                requires(typename C::mapped_type const& obj) {
                                  {
                                    cont.emplace(std::move(key), obj)
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(
                                        std::piecewise_construct,
                                        std::forward_as_tuple(std::move(key)),
                                        std::forward_as_tuple(obj))
                                  } -> std::same_as<typename C::iterator>;
                                };

                   requires not std::movable<typename C::mapped_type> or
                                requires(typename C::mapped_type&& obj) {
                                  {
                                    cont.emplace(std::move(key), std::move(obj))
                                  } -> std::same_as<typename C::iterator>;
                                  {
                                    cont.emplace(
                                        std::piecewise_construct,
                                        std::forward_as_tuple(std::move(key)),
                                        std::forward_as_tuple(std::move(obj)))
                                  } -> std::same_as<typename C::iterator>;
                                };
                 };
  };
};

template <typename C>
concept ordered_map_container =
    map_container<C> and ordered_associative_container<C>;

template <typename C>
concept ordered_unique_map_container =
    unique_map_container<C> and ordered_map_container<C>;

template <typename C>
concept ordered_multiple_map_container =
    multiple_map_container<C> and ordered_map_container<C>;

template <typename C>
concept unordered_map_container =
    map_container<C> and unordered_associative_container<C>;

template <typename C>
concept unordered_unique_map_container =
    unique_map_container<C> and unordered_map_container<C>;

template <typename C>
concept unordered_multiple_map_container =
    multiple_map_container<C> and unordered_map_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept associative_container_of =
    container_of<C, ValueType> and associative_container<C> and
    std::same_as<KeyType, typename C::key_type>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unique_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept multiple_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    multiple_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    ordered_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    ordered_unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept ordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    ordered_multiple_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_associative_container_of =
    associative_container_of<C, ValueType, KeyType> and
    unordered_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_unique_associative_container_of =
    unique_associative_container_of<C, ValueType, KeyType> and
    unordered_unique_associative_container<C>;

template <typename C, typename ValueType, typename KeyType = ValueType>
concept unordered_multiple_associative_container_of =
    multiple_associative_container_of<C, ValueType, KeyType> and
    unordered_multiple_associative_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept map_container_of =
    map_container<C> and std::same_as<KeyType, typename C::key_type> and
    std::same_as<MappedType, typename C::mapped_type>;

template <typename C, typename KeyType, typename MappedType>
concept unique_map_container_of =
    map_container_of<C, KeyType, MappedType> and unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept multiple_map_container_of =
    map_container_of<C, KeyType, MappedType> and multiple_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and ordered_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    ordered_unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept ordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    ordered_multiple_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_map_container_of =
    map_container_of<C, KeyType, MappedType> and unordered_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_unique_map_container_of =
    unique_map_container_of<C, KeyType, MappedType> and
    unordered_unique_map_container<C>;

template <typename C, typename KeyType, typename MappedType>
concept unordered_multiple_map_container_of =
    multiple_map_container_of<C, KeyType, MappedType> and
    unordered_multiple_map_container<C>;

template <typename T, typename U>
concept convertible_to_optional_like =
    utils::traits::is_convertible_to_optional_like<T, U>::value;

template <typename T, typename U>
concept assingable_from_optional_like =
    utils::traits::is_assignable_from_optional_like<T, U>::value;

template <typename T>
concept optional_like = utils::traits::is_optional_like<T>::value;

template <typename T, typename U>
concept assignable_value =
    !utils::traits::is_optional_like<std::decay_t<U>>::value &&
    std::is_constructible_v<T, U> &&
    utils::traits::is_assignable_from_optional_like<T, U>::value &&
    !std::conjunction_v<std::is_scalar<T>, std::is_same<T, std::decay_t<U>>>;

template <typename T, typename U>
concept moveable_assign_from =
    !std::is_same_v<U, T> && utils::traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, U> && std::is_assignable_v<T&, U>;

template <typename T, typename U>
concept copyable_assign_from =
    !std::is_same_v<U, T> && utils::traits::is_converting_ctor<T, U>::value &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

template <typename T, typename U>
concept is_converting_ctor =
    !std::disjunction_v<std::is_reference<U>,
                        std::is_constructible<T, std::optional<U>&>,
                        std::is_constructible<T, const std::optional<U>&>,
                        std::is_constructible<T, std::optional<U>&&>,
                        std::is_constructible<T, const std::optional<U>&&>,
                        std::is_convertible<std::optional<U>&, T>,
                        std::is_convertible<const std::optional<U>&, T>,
                        std::is_convertible<std::optional<U>&&, T>,
                        std::is_convertible<const std::optional<U>&&, T>>;

template <typename T, typename U>
concept is_converting_assign =
    is_converting_ctor<T, U> &&
    !std::disjunction_v<std::is_assignable<T&, std::optional<U>&>,
                        std::is_assignable<T&, const std::optional<U>&>,
                        std::is_assignable<T&, std::optional<U>&&>,
                        std::is_assignable<T&, const std::optional<U>&&>>;

template <typename T, typename U>
concept is_implicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_convertible_v<const U&, T>;

template <typename T, typename U>
concept is_explicitly_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, const U&> && !std::is_convertible_v<const U&, T>;

template <typename T, typename U>
concept is_implicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_explicitly_move_convertible =
    !std::is_same_v<U, T> && is_converting_ctor<T, U> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_directly_constructible =
    !traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_explicitly_directly_constructible =
    !traits::is_optional_v<std::decay_t<U>> &&
    std::is_constructible_v<T, U&&> && !std::is_convertible_v<U&&, T>;

template <typename T, typename U>
concept is_copy_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign<T, U> &&
    std::is_constructible_v<T, const U&> && std::is_assignable_v<T&, const U&>;

template <typename T, typename U>
concept is_move_assignable_from =
    !std::is_same_v<U, T> && is_converting_assign<T, U> &&
    std::is_constructible_v<T, U&&> && std::is_assignable_v<T&, U&&>;
}  // namespace core::meta::concepts