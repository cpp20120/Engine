#pragma once

#include <concepts>
#include <coroutine>
#include <optional>
#include <type_traits>

#include "../dynamic_optional.hpp"
#include "../meta_functions/meta_functions.hpp"

namespace core::meta::type_traits {

/**
 * @brief Constant for the mathematical constant e (Euler's number).
 * @tparam T The type of the constant.
 */
template <typename T>
inline constexpr T e = T(2.7182818284590452353);

/**
 * @brief Constant for the mathematical constant pi.
 * @tparam T The type of the constant.
 */
template <typename T>
inline constexpr T pi = T(3.1415926535897932385);

/**
 * @brief Template alias for creating a compile-time constant.
 * @tparam N The value of the constant.
 * @tparam T The type of the constant.
 */
template <auto N, typename T = std::remove_cvref_t<decltype(N)>>
using c_ = std::integral_constant<T, N>;

/**
 * @brief Compile-time constant for 0.
 */
using c_0 = c_<0>;

/**
 * @brief Compile-time constant for 1.
 */
using c_1 = c_<1>;

/**
 * @brief Compile-time constant for 2.
 */
using c_2 = c_<2>;

/**
 * @brief Compile-time constant for 3.
 */
using c_3 = c_<3>;

/**
 * @brief Compile-time constant for 4.
 */
using c_4 = c_<4>;

/**
 * @brief Compile-time constant for 5.
 */
using c_5 = c_<5>;

/**
 * @brief Compile-time constant for 6.
 */
using c_6 = c_<6>;

/**
 * @brief Compile-time constant for 7.
 */
using c_7 = c_<7>;

/**
 * @brief Compile-time constant for 8.
 */
using c_8 = c_<8>;

/**
 * @brief Compile-time constant for 9.
 */
using c_9 = c_<9>;

/**
 * @brief Template alias for creating a compile-time boolean constant.
 * @tparam N The boolean value of the constant.
 */
template <auto N>
using bool_ = std::bool_constant<N>;

/**
 * @brief Template alias for creating an index type.
 * @tparam N The value of the index.
 */
template <size_t N>
using index_t = c_<N, size_t>;

/**
 * @brief Compile-time index constant.
 * @tparam N The value of the index.
 */
template <size_t N>
constexpr index_t<N> index = {};

/**
 * @brief Template alias for creating a compile-time constant of a given value.
 * @tparam v The value of the constant.
 */
template <auto v>
using constant_t = c_<v, std::decay_t<decltype(v)>>;

/**
 * @brief Compile-time constant of a given value.
 * @tparam v The value of the constant.
 */
template <auto v>
constexpr constant_t<v> constant = {};

/**
 * @brief Template alias for creating a tuple type.
 * @tparam Args The types of the tuple elements.
 */
template <typename... Args>
using tuple_t = std::tuple<Args...>;

/**
 * @brief Template alias for creating an index sequence.
 * @tparam N The indices of the sequence.
 */
template <auto... N>
using is = std::index_sequence<N...>;

/**
 * @brief Extracts the value from a type with a static value member.
 * @tparam T The type with a static value member.
 */
template <typename T>
inline constexpr auto typev = T::value;

/**
 * @brief Template alias for extracting the value type from a type.
 * @tparam T The type with a value_type member.
 */
template <typename T>
using value_t = typename T::value_type;

/**
 * @brief Negates a boolean value.
 * @tparam T The type with a boolean value.
 */
template <typename T>
inline constexpr auto negav = std::negation_v<T>;

/**
 * @brief Template struct to determine if a type has a nested type.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct typeof {
  using type = T;
  static constexpr auto value = 0;
};

/**
 * @brief Specialization of typeof for types with a nested type.
 * @tparam T The type to check.
 */
template <typename T>
struct typeof<T, std::void_t<typename T::type>> {
  using type = typename T::type;
  static constexpr auto value = 1;
};

/**
 * @brief Template alias for extracting the nested type from a type.
 * @tparam T The type to check.
 */
template <typename T>
using typeof_t = typename typeof<T>::type;

/**
 * @brief Extracts the value from a type with a nested type.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto typeof_v = typev<typeof_t<T>>;

/**
 * @brief Template alias for selecting a type based on a boolean condition.
 * @tparam B The boolean condition.
 * @tparam T The type to select if the condition is true.
 * @tparam U The type to select if the condition is false.
 */
template <bool B, typename T, typename U>
using type_if = typeof_t<std::conditional_t<B, T, U>>;

/**
 * @brief Extracts the value from a type selected based on a boolean condition.
 * @tparam B The boolean condition.
 * @tparam T The type to select if the condition is true.
 * @tparam U The type to select if the condition is false.
 */
template <bool B, typename T, typename U>
inline constexpr auto type_if_v = typev<type_if<B, T, U>>;

/**
 * @brief Extracts the value from a type selected based on a boolean condition.
 * @tparam B The boolean condition.
 * @tparam T The type to select if the condition is true.
 * @tparam U The type to select if the condition is false.
 */
template <bool B, typename T, typename U>
inline constexpr auto value_if = typev<std::conditional_t<B, T, U>>;

/**
 * @brief Template alias for selecting a type based on a type condition.
 * @tparam T The type condition.
 * @tparam U The type to select if the condition is true.
 * @tparam V The type to select if the condition is false.
 */
template <typename T, typename U, typename V>
using conditional_of = std::conditional_t<typev<T>, U, V>;

/**
 * @brief Template struct to determine if a type has a nested type.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct has_type : std::false_type {};

/**
 * @brief Specialization of has_type for types with a nested type.
 * @tparam T The type to check.
 */
template <typename T>
struct has_type<T, std::void_t<typename T::type>> : std::true_type {};

/**
 * @brief Extracts the value from a type with a nested type.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto has_type_v = typev<has_type<T>>;

/**
 * @brief Template struct to determine if a type has a nested value_type.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct has_value_type : std::false_type {
  using value_type = int;
};

/**
 * @brief Specialization of has_value_type for types with a nested value_type.
 * @tparam T The type to check.
 */
template <typename T>
struct has_value_type<T, std::void_t<value_t<T>>> : std::true_type {
  using value_type = value_t<T>;
};

/**
 * @brief Template alias for extracting the nested value_type from a type.
 * @tparam T The type to check.
 */
template <typename T>
using has_value_type_t = value_t<has_value_type<T>>;

/**
 * @brief Extracts the value from a type with a nested value_type.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto has_value_type_v = typev<has_value_type<T>>;

/**
 * @brief Template struct to determine if a type has a custom operator new.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct has_new : std::false_type {};

/**
 * @brief Specialization of has_new for types with a custom operator new.
 * @tparam T The type to check.
 */
template <typename T>
struct has_new<T, std::void_t<decltype(T::operator new(0))>> : std::true_type {
};

/**
 * @brief Extracts the value from a type with a custom operator new.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto has_new_v = typev<has_new<T>>;

/**
 * @brief Template struct to determine if a type has a custom operator delete.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct has_delete : std::false_type {};

/**
 * @brief Specialization of has_delete for types with a custom operator delete.
 * @tparam T The type to check.
 */
template <typename T>
struct has_delete<T, std::void_t<decltype(T::operator delete(nullptr))>>
    : std::true_type {};

/**
 * @brief Extracts the value from a type with a custom operator delete.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto has_delete_v = typev<has_delete<T>>;

/**
 * @brief Template struct to determine if a type is complete.
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct is_type_complete : std::false_type {};

/**
 * @brief Specialization of is_type_complete for complete types.
 * @tparam T The type to check.
 */
template <typename T>
struct is_type_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {
};

/**
 * @brief Extracts the value from a type that is complete.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto is_type_complete_v = typev<is_type_complete<T>>;

/**
 * @brief Template struct to determine if a type is a base template of another
 * type.
 * @tparam B The base template to check.
 * @tparam T The type to check.
 */
template <template <typename...> typename B, typename T,
          typename = std::void_t<>>
struct is_base_template_of : std::false_type {};

/**
 * @brief Specialization of is_base_template_of for types that are base
 * templates.
 * @tparam B The base template to check.
 * @tparam T The type to check.
 */
template <template <typename...> typename B, typename T>
struct is_base_template_of<
    B, T, std::void_t<decltype([]<typename... Args>(B<Args...> *) {
    }(std::declval<T *>()))>> : std::true_type {};

/**
 * @brief Extracts the value from a type that is a base template of another
 * type.
 * @tparam B The base template to check.
 * @tparam T The type to check.
 */
template <template <typename...> typename B, typename T>
inline constexpr auto is_base_template_of_v = typev<is_base_template_of<B, T>>;

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The type to check.
 * @return false for non-template types.
 */
template <typename T>
constexpr bool is_template() {
  return false;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <auto...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <typename...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <typename, auto...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <template <auto...> typename...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <template <typename...> typename...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Function to determine if a type is a template.
 * @tparam T The template to check.
 * @return true for template types.
 */
template <template <template <typename, auto...> typename...> typename T>
constexpr bool is_template() {
  return true;
}

/**
 * @brief Template struct for creating a pair of types.
 * @tparam T The first type.
 * @tparam U The second type.
 */
template <typename T, typename U>
struct pair_t {
  using first = T;
  using second = U;
};

/**
 * @brief Template alias for extracting the first type from a pair.
 * @tparam T The pair type.
 */
template <typename T>
using first_t = typename T::first;

/**
 * @brief Template alias for extracting the second type from a pair.
 * @tparam T The pair type.
 */
template <typename T>
using second_t = typename T::second;

/**
 * @brief Template struct for creating a pair of values.
 * @tparam p The first value.
 * @tparam q The second value.
 */
template <int p, int q>
struct pair_v {
  static constexpr auto first = p;
  static constexpr auto second = q;
};

/**
 * @brief Extracts the first value from a pair.
 * @tparam T The pair type.
 */
template <typename T>
inline constexpr auto first_v = T::first;

/**
 * @brief Extracts the second value from a pair.
 * @tparam T The pair type.
 */
template <typename T>
inline constexpr auto second_v = T::second;

/**
 * @brief Computes the difference between the second and first values of a pair.
 * @tparam T The pair type.
 */
template <typename T>
inline constexpr auto pair_diff = second_v<T> - first_v<T>;

/**
 * @brief Template struct for creating a triple of values.
 * @tparam p The first value.
 * @tparam q The second value.
 * @tparam T The type.
 */
template <auto p, auto q, typename T>
struct triple : std::type_identity<T> {
  static constexpr auto first = p;
  static constexpr auto second = q;
};

/**
 * @brief Template struct for creating an identity type.
 * @tparam N The index.
 * @tparam T The type.
 */
template <int N, typename T>
struct identity {
  using type = T;
};

/**
 * @brief Template alias for extracting the type from an identity.
 * @tparam N The index.
 * @tparam T The type.
 */
template <int N, typename T>
using identity_t = typeof_t<identity<N, T>>;

/**
 * @brief Function to ignore a value.
 * @tparam N The index.
 * @tparam T The type of the value.
 * @param t The value to ignore.
 * @return The value.
 */
template <auto N, typename T>
constexpr decltype(auto) ignore(T &&t) {
  return std::forward<T>(t);
}

/**
 * @brief Template struct for creating a wrapper type.
 * @tparam N The index.
 * @tparam T The type.
 */
template <auto N, typename T>
struct wrapper : wrapper<N - 1, std::type_identity<T>> {};

/**
 * @brief Specialization of wrapper for index 0.
 * @tparam T The type.
 */
template <typename T>
struct wrapper<0, T> : std::type_identity<T> {};

/**
 * @brief Template alias for extracting the type from a wrapper.
 * @tparam N The index.
 * @tparam T The type.
 */
template <auto N, typename T>
using wrapper_t = typeof_t<wrapper<N, T>>;

/**
 * @brief Template struct for creating an index type.
 * @tparam N The index.
 * @tparam T The type.
 */
template <auto N, typename T>
struct index_type : std::type_identity<T> {
  static constexpr auto value = N;
};

/**
 * @brief Template alias for creating an upper index type.
 * @tparam N The index.
 * @tparam T The type.
 */
template <auto N, typename T>
using index_upper = wrapper_t<1, index_type<N, T>>;

/**
 * @brief Template struct for creating an alias type.
 * @tparam T The type.
 * @tparam Args The additional types.
 */
template <typename T, typename...>
struct alias {
  using type = T;
};

/**
 * @brief Template alias for extracting the type from an alias.
 * @tparam Args The types.
 */
template <typename... Args>
using alias_t = typeof_t<alias<Args...>>;

/**
 * @brief Template struct for wrapping types in a template.
 * @tparam F The template to wrap in.
 * @tparam Args The types to wrap.
 */
template <template <typename...> typename F, typename... Args>
struct wrapin {
  using type = tuple_t<F<Args>...>;
};

/**
 * @brief Template alias for extracting the type from a wrapin.
 * @tparam F The template to wrap in.
 * @tparam Args The types to wrap.
 */
template <template <typename...> typename F, typename... Args>
using wrapin_t = typeof_t<wrapin<F, Args...>>;

/**
 * @brief Template alias for wrapping a type in a template based on a boolean
 * condition.
 * @tparam B The boolean condition.
 * @tparam F The template to wrap in.
 * @tparam T The type to wrap.
 */
template <bool B, template <typename...> typename F, typename T>
using wrapin_if = std::conditional_t<B, F<T>, T>;

/**
 * @brief Template struct to determine if a type is contained in a list of
 * types.
 * @tparam Args The types to check.
 */
template <typename... Args>
struct contains : std::false_type {};

/**
 * @brief Specialization of contains for types that are contained.
 * @tparam T The type to check.
 * @tparam Args The types to check.
 */
template <typename T, typename... Args>
struct contains<T, Args...> : bool_<(std::is_same_v<T, Args> || ...)> {};

/**
 * @brief Extracts the value from a type that is contained in a list of types.
 * @tparam Args The types to check.
 */
template <typename...>
inline constexpr auto contains_v = std::false_type{};

/**
 * @brief Extracts the value from a type that is contained in a list of types.
 * @tparam T The type to check.
 * @tparam Args The types to check.
 */
template <typename T, typename... Args>
inline constexpr auto contains_v<T, Args...> = (std::is_same_v<T, Args> || ...);

/**
 * @brief Template struct to determine if a list of values comprises a specific
 * value.
 * @tparam values The values to check.
 */
template <auto... values>
struct comprise : std::false_type {};

/**
 * @brief Specialization of comprise for values that are comprised.
 * @tparam value The value to check.
 * @tparam values The values to check.
 */
template <auto value, auto... values>
struct comprise<value, values...> : bool_<((value == values) || ...)> {};

/**
 * @brief Extracts the value from a type that comprises a specific value.
 * @tparam values The values to check.
 */
template <auto...>
inline constexpr auto comprise_v = std::false_type{};

/**
 * @brief Extracts the value from a type that comprises a specific value.
 * @tparam value The value to check.
 * @tparam values The values to check.
 */
template <auto value, auto... values>
inline constexpr auto comprise_v<value, values...> = ((value == values) || ...);

/**
 * @brief Template struct to determine if a type exists in a list of types.
 * @tparam B The base type.
 * @tparam Args The types to check.
 */
template <typename B, typename...>
struct exists_type : B {};

/**
 * @brief Specialization of exists_type for types that exist.
 * @tparam B The base type.
 * @tparam T The type to check.
 * @tparam Args The types to check.
 */
template <typename B, typename T, typename... Args>
struct exists_type<B, T, Args...>
    : std::conditional_t<contains_v<T, Args...>, std::negation<B>,
                         exists_type<B, Args...>> {};

/**
 * @brief Template alias for extracting the type from an exists_type.
 * @tparam B The base type.
 * @tparam Args The types to check.
 */
template <typename B, typename... Args>
using exists_type_t = typeof_t<exists_type<B, Args...>>;

/**
 * @brief Extracts the value from a type that exists in a list of types.
 * @tparam B The base type.
 * @tparam Args The types to check.
 */
template <typename B, typename... Args>
inline constexpr auto exists_type_v = typev<exists_type_t<B, Args...>>;

/**
 * @brief Template alias for determining if a list of types is unique.
 * @tparam Args The types to check.
 */
template <typename... Args>
using is_unique_type = exists_type<std::true_type, Args...>;

/**
 * @brief Extracts the value from a type that is unique in a list of types.
 * @tparam Args The types to check.
 */
template <typename...>
inline constexpr auto is_unique_type_v = std::true_type{};

/**
 * @brief Extracts the value from a type that is unique in a list of types.
 * @tparam T The type to check.
 * @tparam Args The types to check.
 */
template <typename T, typename... Args>
inline constexpr auto is_unique_type_v<T, Args...> =
    !contains_v<T, Args...> && is_unique_type_v<Args...>;

/**
 * @brief Template alias for determining if a list of types has duplicates.
 * @tparam Args The types to check.
 */
template <typename... Args>
using has_duplicates_type = exists_type<std::false_type, Args...>;

/**
 * @brief Extracts the value from a type that has duplicates in a list of types.
 * @tparam Args The types to check.
 */
template <typename...>
inline constexpr auto has_duplicates_type_v = std::false_type{};

/**
 * @brief Extracts the value from a type that has duplicates in a list of types.
 * @tparam T The type to check.
 * @tparam Args The types to check.
 */
template <typename T, typename... Args>
inline constexpr auto has_duplicates_type_v<T, Args...> =
    contains_v<T, Args...> || has_duplicates_type_v<Args...>;

/**
 * @brief Template struct to determine if a list of values is unique.
 * @tparam values The values to check.
 */
template <auto... values>
using is_unique_value = exists_value<std::true_type, values...>;

/**
 * @brief Extracts the value from a type that is unique in a list of values.
 * @tparam values The values to check.
 */
template <auto...>
inline constexpr auto is_unique_value_v = std::true_type{};

/**
 * @brief Extracts the value from a type that is unique in a list of values.
 * @tparam value The value to check.
 * @tparam values The values to check.
 */
template <auto value, auto... values>
inline constexpr auto is_unique_value_v<value, values...> =
    negav<comprise<value, values...>> && is_unique_value_v<values...>;

/**
 * @brief Template alias for determining if a list of values has duplicates.
 * @tparam values The values to check.
 */
template <auto... values>
using has_duplicates_value = exists_value<std::false_type, values...>;

/**
 * @brief Extracts the value from a type that has duplicates in a list of
 * values.
 * @tparam values The values to check.
 */
template <auto...>
inline constexpr auto has_duplicates_value_v = std::false_type{};

/**
 * @brief Extracts the value from a type that has duplicates in a list of
 * values.
 * @tparam value The value to check.
 * @tparam values The values to check.
 */
template <auto value, auto... values>
inline constexpr auto has_duplicates_value_v<value, values...> =
    typev<comprise<value, values...>> || has_duplicates_value_v<values...>;

/**
 * @brief Template struct to determine if a type exists based on a boolean
 * condition.
 * @tparam B The boolean condition.
 * @tparam T The type to check.
 */
template <bool B, typename T>
struct exists;

/**
 * @brief Specialization of exists for template types with a boolean condition.
 * @tparam T The template to check.
 * @tparam Args The types to check.
 */
template <template <typename...> typename T, typename... Args>
struct exists<true, T<Args...>> {
  using type = is_unique_type<Args...>;
};

/**
 * @brief Specialization of exists for template types with a boolean condition.
 * @tparam T The template to check.
 * @tparam U The type to check.
 * @tparam values The values to check.
 */
template <template <typename, auto...> typename T, typename U, auto... values>
struct exists<true, T<U, values...>> {
  using type = is_unique_value<values...>;
};

/**
 * @brief Specialization of exists for template types with a boolean condition.
 * @tparam T The template to check.
 * @tparam Args The types to check.
 */
template <template <typename...> typename T, typename... Args>
struct exists<false, T<Args...>> {
  using type = has_duplicates_type<Args...>;
};

/**
 * @brief Specialization of exists for template types with a boolean condition.
 * @tparam T The template to check.
 * @tparam U The type to check.
 * @tparam values The values to check.
 */
template <template <typename, auto...> typename T, typename U, auto... values>
struct exists<false, T<U, values...>> {
  using type = has_duplicates_value<values...>;
};

/**
 * @brief Template alias for extracting the type from an exists.
 * @tparam B The boolean condition.
 * @tparam T The type to check.
 */
template <bool B, typename T>
using exists_t = typeof_t<exists<B, T>>;

/**
 * @brief Extracts the value from a type that exists based on a boolean
 * condition.
 * @tparam B The boolean condition.
 * @tparam T The type to check.
 */
template <bool B, typename T>
inline constexpr auto exists_v = typev<exists_t<B, T>>;

/**
 * @brief Template alias for determining if a type is unique.
 * @tparam T The type to check.
 */
template <typename T>
using is_unique = exists<true, T>;

/**
 * @brief Template alias for extracting the type from an is_unique.
 * @tparam T The type to check.
 */
template <typename T>
using is_unique_t = typeof_t<is_unique<T>>;

/**
 * @brief Extracts the value from a type that is unique.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto is_unique_v = typev<is_unique_t<T>>;

/**
 * @brief Template alias for determining if a type has duplicates.
 * @tparam T The type to check.
 */
template <typename T>
using has_duplicates = exists<false, T>;

/**
 * @brief Template alias for extracting the type from a has_duplicates.
 * @tparam T The type to check.
 */
template <typename T>
using has_duplicates_t = typeof_t<has_duplicates<T>>;

/**
 * @brief Extracts the value from a type that has duplicates.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto has_duplicates_v = typev<has_duplicates_t<T>>;

/**
 * @brief Template alias for ternary conditional type selection.
 * @tparam A The first boolean condition.
 * @tparam B The second boolean condition.
 * @tparam X The type to select if both conditions are true.
 * @tparam Y The type to select if the first condition is true and the second is
 * false.
 * @tparam Z The type to select if the first condition is false.
 */
template <bool A, bool B, typename X, typename Y, typename Z>
using ternary_conditional = std::conditional<A, std::conditional_t<B, X, Y>, Z>;

/**
 * @brief Template alias for extracting the type from a ternary_conditional.
 * @tparam A The first boolean condition.
 * @tparam B The second boolean condition.
 * @tparam X The type to select if both conditions are true.
 * @tparam Y The type to select if the first condition is true and the second is
 * false.
 * @tparam Z The type to select if the first condition is false.
 */
template <bool A, bool B, typename X, typename Y, typename Z>
using ternary_conditional_t = typeof_t<ternary_conditional<A, B, X, Y, Z>>;

/**
 * @brief Template struct for inheriting from multiple types.
 * @tparam Args The types to inherit from.
 */
template <typename... Args>
struct inherit : Args... {};

/**
 * @brief Template struct to determine if a type is inheritable.
 * @tparam T The type to check.
 */
template <typename T, typename U = std::void_t<>>
struct is_inheritable : std::false_type {};

/**
 * @brief Specialization of is_inheritable for inheritable types.
 * @tparam T The type to check.
 */
template <typename T>
struct is_inheritable<T, std::void_t<inherit<T>>> : std::true_type {};

/**
 * @brief Extracts the value from a type that is inheritable.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto is_inheritable_v = typev<is_inheritable<T>>;

/**
 * @brief Template struct to determine if a pack of types is inheritable.
 * @tparam Args The types to check.
 */
template <typename... Args>
struct is_inheritable_pack : bool_<(is_inheritable_v<Args> && ...)> {};

/**
 * @brief Extracts the value from a type that is inheritable in a pack of types.
 * @tparam Args The types to check.
 */
template <typename... Args>
inline constexpr auto is_inheritable_pack_v =
    typev<is_inheritable_pack<Args...>>;

/**
 * @brief Template struct to determine if a type is instantiable.
 * @tparam T The type to check.
 */
template <typename T>
struct is_instantiable : std::negation<std::is_abstract<T>> {};

/**
 * @brief Extracts the value from a type that is instantiable.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr auto is_instantiable_v = typev<is_instantiable<T>>;

/**
 * @brief Identity type that holds the given type T
 * @tparam T The type to hold
 */
template <typename T>
struct type_identity {
  using type = T;
};

/**
 * @brief Integral constant representing a size_t index
 * @tparam idx The index value
 */
template <std::size_t idx>
using index_constant = std::integral_constant<std::size_t, idx>;

/**
 * @brief Pair of size_t indices
 * @tparam t_first First index
 * @tparam t_second Second index
 */
template <std::size_t t_first, std::size_t t_second>
struct index_pair {
  constexpr inline static std::size_t first{t_first};
  constexpr inline static std::size_t second{t_second};
};

/**
 * @brief Computes the sum type of given types
 * @tparam Ts Types to compute sum for
 */
template <typename... Ts>
using sum_type = type_identity<decltype((... + std::declval<Ts>()))>;

/**
 * @brief Helper alias for sum_type
 * @tparam Ts Types to compute sum for
 */
template <typename... Ts>
using sum_type_t = typename sum_type<Ts...>::type;

/**
 * @brief Removes const, volatile and reference qualifiers from a type
 * @tparam T Type to remove qualifiers from
 */
template <typename T>
using remove_cvref = std::remove_cv<std::remove_reference_t<T>>;

/**
 * @brief Helper alias for remove_cvref
 * @tparam T Type to remove qualifiers from
 */
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

/**
 * @brief Checks if type T is present in the given pack
 * @tparam T Type to check for
 * @tparam Pack Pack of types to search in
 */
template <typename T, typename... Pack>
struct is_type_in_pack : std::bool_constant<(std::is_same_v<T, Pack> || ...)> {
};

/**
 * @brief Specialization for empty pack
 * @tparam T Type to check for
 */
template <typename T>
struct is_type_in_pack<T> : std::false_type {};

/**
 * @brief Helper variable template for is_type_in_pack
 * @tparam T Type to check for
 * @tparam Pack Pack of types to search in
 */
template <typename T, typename... Pack>
constexpr inline bool is_type_in_pack_v = is_type_in_pack<T, Pack...>::value;

/**
 * @brief Extracts the first type from a pack
 * @tparam Pack Pack of types
 */
template <typename... Pack>
struct peel_first : type_identity<void> {};

/**
 * @brief Specialization for non-empty pack
 * @tparam First First type in pack
 * @tparam Pack Rest of types
 */
template <typename First, typename... Pack>
struct peel_first<First, Pack...> : type_identity<First> {};

/**
 * @brief Helper alias for peel_first
 * @tparam Pack Pack of types
 */
template <typename... Pack>
using peel_first_t = typename peel_first<Pack...>::type;

/**
 * @brief Extracts the last type from a pack
 * @tparam Pack Pack of types
 */
template <typename... Pack>
struct peel_last : type_identity<void> {};

/**
 * @brief Recursive case for peel_last
 * @tparam First First type in pack
 * @tparam Pack Rest of types
 */
template <typename First, typename... Pack>
struct peel_last<First, Pack...>
    : type_identity<typename peel_last<Pack...>::type> {};

/**
 * @brief Base case for peel_last
 * @tparam Last Last type in pack
 */
template <typename Last>
struct peel_last<Last> : type_identity<Last> {};

/**
 * @brief Helper alias for peel_last
 * @tparam Pack Pack of types
 */
template <typename... Pack>
using peel_last_t = typename peel_last<Pack...>::type;

/**
 * @brief Checks if all types in a pack are the same
 * @tparam Pack Pack of types to check
 */
template <typename... Pack>
struct is_pack_uniform {};

/**
 * @brief Specialization for non-empty pack
 * @tparam T First type in pack
 * @tparam Pack Rest of types to compare
 */
template <typename T, typename... Pack>
struct is_pack_uniform<T, Pack...>
    : std::bool_constant<(std::is_same_v<T, Pack> && ...)> {};

/**
 * @brief Helper variable template for is_pack_uniform
 * @tparam Pack Pack of types to check
 */
template <typename... Pack>
constexpr inline bool is_pack_uniform_v = is_pack_uniform<Pack...>::value;

/**
 * @brief Checks if pack contains only the given type T
 * @tparam T Type to check for
 * @tparam Pack Pack of types
 */
template <typename T, typename... Pack>
struct is_pack_only : std::conjunction<is_pack_uniform<Pack...>,
                                       std::is_same<T, peel_first_t<Pack...>>> {
};

/**
 * @brief Specialization for empty pack
 * @tparam T Type to check for
 */
template <typename T>
struct is_pack_only<T> : std::false_type {};

/**
 * @brief Helper variable template for is_pack_only
 * @tparam T Type to check for
 * @tparam Pack Pack of types
 */
template <typename T, typename... Pack>
constexpr inline bool is_pack_only_v = is_pack_only<T, Pack...>::value;

/**
 * @brief Checks if a type is iterable (has begin() and end())
 * @tparam T Type to check
 */
template <typename T, typename = void>
struct is_iterable : std::false_type {};

/**
 * @brief Specialization for iterable types
 * @tparam T Iterable type
 */
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T &>())),
                                  decltype(std::end(std::declval<T &>()))>>
    : std::true_type {};

/**
 * @brief Helper variable template for is_iterable
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_iterable_v = is_iterable<T>::value;

/**
 * @brief Checks if a type is an iterator (has * and ++ operators)
 * @tparam T Type to check
 */
template <typename T, typename = void>
struct is_iterator : std::false_type {};

/**
 * @brief Specialization for iterator types
 * @tparam T Iterator type
 */
template <typename T>
struct is_iterator<T, std::void_t<decltype(*std::declval<T>()),
                                  decltype(++std::declval<T &>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for is_iterator
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_iterator_v = is_iterator<T>::value;

/**
 * @brief Checks if an iterator is a forward iterator
 * @tparam T Iterator type to check
 */
template <typename T>
struct is_forward
    : std::is_base_of<std::forward_iterator_tag,
                      typename std::iterator_traits<T>::iterator_category> {};

/**
 * @brief Helper variable template for is_forward
 * @tparam T Iterator type to check
 */
template <typename T>
constexpr inline bool is_forward_v = is_forward<T>::value;

/**
 * @brief Checks if an iterator is a bidirectional iterator
 * @tparam T Iterator type to check
 */
template <typename T>
struct is_bidirectional
    : std::is_base_of<std::bidirectional_iterator_tag,
                      typename std::iterator_traits<T>::iterator_category> {};

/**
 * @brief Helper variable template for is_bidirectional
 * @tparam T Iterator type to check
 */
template <typename T>
constexpr inline bool is_bidirectional_v = is_bidirectional<T>::value;

/**
 * @brief Checks if an iterator is a random access iterator
 * @tparam T Iterator type to check
 */
template <typename T>
struct is_random_access
    : std::is_base_of<std::random_access_iterator_tag,
                      typename std::iterator_traits<T>::iterator_category> {};

/**
 * @brief Helper variable template for is_random_access
 * @tparam T Iterator type to check
 */
template <typename T>
constexpr inline bool is_random_access_v = is_random_access<T>::value;

namespace impl {
/**
 * @brief Implementation helper for is_tuple
 * @tparam Ts Types to check
 */
template <typename... Ts>
struct is_tuple_impl : std::false_type {};

/**
 * @brief Specialization for std::tuple
 * @tparam Ts Tuple element types
 */
template <typename... Ts>
struct is_tuple_impl<std::tuple<Ts...>> : std::true_type {};
}  // namespace impl

/**
 * @brief Checks if a type is a std::tuple
 * @tparam T Type to check
 */
template <typename T>
struct is_tuple : impl::is_tuple_impl<remove_cvref_t<T>> {};

/**
 * @brief Helper variable template for is_tuple
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_tuple_v = is_tuple<T>::value;

namespace impl {
/**
 * @brief Implementation helper for is_pair
 * @tparam T Type to check
 */
template <typename T>
struct is_pair_impl : std::false_type {};

/**
 * @brief Specialization for std::pair
 * @tparam First First element type
 * @tparam Second Second element type
 */
template <typename First, typename Second>
struct is_pair_impl<std::pair<First, Second>> : std::true_type {};
}  // namespace impl

/**
 * @brief Checks if a type is a std::pair
 * @tparam T Type to check
 */
template <typename T>
struct is_pair : impl::is_pair_impl<remove_cvref_t<T>> {};

/**
 * @brief Helper variable template for is_pair
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_pair_v = is_pair<T>::value;

/**
 * @brief Checks if a type is a smart pointer
 * @tparam T Type to check
 */
template <typename T>
struct is_smart_pointer : std::false_type {};

/**
 * @brief Specialization for std::unique_ptr
 * @tparam T Element type
 */
template <typename T>
struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {};

/**
 * @brief Specialization for std::unique_ptr with custom deleter
 * @tparam T Element type
 * @tparam U Deleter type
 */
template <typename T, typename U>
struct is_smart_pointer<std::unique_ptr<T, U>> : std::true_type {};

/**
 * @brief Specialization for std::shared_ptr
 * @tparam T Element type
 */
template <typename T>
struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};

/**
 * @brief Specialization for std::weak_ptr
 * @tparam T Element type
 */
template <typename T>
struct is_smart_pointer<std::weak_ptr<T>> : std::true_type {};

/**
 * @brief Helper variable template for is_smart_pointer
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_smart_pointer_v = is_smart_pointer<T>::value;

/**
 * @brief Checks if a type can be printed to std::ostream
 * @tparam T Type to check
 */
template <typename T, typename = void>
struct is_printable : std::false_type {};

/**
 * @brief Specialization for printable types
 * @tparam T Printable type
 */
template <typename T>
struct is_printable<T, std::void_t<decltype(std::declval<std::ostream &>()
                                            << std::declval<T>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for is_printable
 * @tparam T Type to check
 */
template <typename T>
constexpr inline bool is_printable_v = is_printable<T>::value;

/**
 * @brief Checks if two types are equality comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_equality_comparable : std::false_type {};

/**
 * @brief Specialization for equality comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_equality_comparable<
    T, U, std::void_t<decltype(std::declval<T>() == std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_equality_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_equality_comparable_v =
    are_equality_comparable<T, U>::value;

/**
 * @brief Checks if two types are inequality comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_inequality_comparable : std::false_type {};

/**
 * @brief Specialization for inequality comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_inequality_comparable<
    T, U, std::void_t<decltype(std::declval<T>() != std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_inequality_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_inequality_comparable_v =
    are_inequality_comparable<T, U>::value;

/**
 * @brief Checks if two types are less comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_less_comparable : std::false_type {};

/**
 * @brief Specialization for less comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_less_comparable<
    T, U, std::void_t<decltype(std::declval<T>() < std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_less_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_less_comparable_v = are_less_comparable<T, U>::value;

/**
 * @brief Checks if two types are less or equal comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_less_eq_comparable : std::false_type {};

/**
 * @brief Specialization for less or equal comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_less_eq_comparable<
    T, U, std::void_t<decltype(std::declval<T>() <= std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_less_eq_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_less_eq_comparable_v =
    are_less_eq_comparable<T, U>::value;

/**
 * @brief Checks if two types are greater comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_greater_comparable : std::false_type {};

/**
 * @brief Specialization for greater comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_greater_comparable<
    T, U, std::void_t<decltype(std::declval<T>() > std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_greater_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_greater_comparable_v =
    are_greater_comparable<T, U>::value;

/**
 * @brief Checks if two types are greater or equal comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U, typename = void>
struct are_greater_eq_comparable : std::false_type {};

/**
 * @brief Specialization for greater or equal comparable types
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
struct are_greater_eq_comparable<
    T, U, std::void_t<decltype(std::declval<T>() >= std::declval<U>())>>
    : std::true_type {};

/**
 * @brief Helper variable template for are_greater_eq_comparable
 * @tparam T First type
 * @tparam U Second type
 */
template <typename T, typename U>
constexpr inline bool are_greater_eq_comparable_v =
    are_greater_eq_comparable<T, U>::value;

/**
 * @brief Makes a const lvalue reference type
 * @tparam T Type to convert
 */
template <typename T>
struct make_const_ref
    : type_identity<
          std::add_lvalue_reference_t<std::add_const_t<remove_cvref_t<T>>>> {};

/**
 * @brief Helper alias for make_const_ref
 * @tparam T Type to convert
 */
template <typename T>
using make_const_ref_t = typename make_const_ref<T>::type;

/**
 * @brief Checks if a type is same as T
 * @tparam T Type to compare against
 */
template <typename T>
struct is_same_as {
  /**
   * @brief Functor to check type equality
   * @tparam U Type to compare
   */
  template <typename U>
  struct func : std::is_same<T, U> {};

  /**
   * @brief Helper variable template for func
   * @tparam U Type to compare
   */
  template <typename U>
  constexpr inline static bool func_v = func<U>::value;
};

/**
 * @brief Composes multiple predicates with conjunction
 * @tparam PREDS Predicates to compose
 */
template <template <typename> typename... PREDS>
struct conjunction_compose {
  /**
   * @brief Applies conjunction to predicates for type T
   * @tparam T Type to check
   */
  template <typename T>
  using func = std::conjunction<PREDS<T>...>;

  /**
   * @brief Helper variable template for func
   * @tparam T Type to check
   */
  template <typename T>
  constexpr inline static bool func_v = func<T>::value;
};

/**
 * @brief Composes multiple predicates with disjunction
 * @tparam PREDS Predicates to compose
 */
template <template <typename> typename... PREDS>
struct disjunction_compose {
  /**
   * @brief Applies disjunction to predicates for type T
   * @tparam T Type to check
   */
  template <typename T>
  using func = std::disjunction<PREDS<T>...>;

  /**
   * @brief Helper variable template for func
   * @tparam T Type to check
   */
  template <typename T>
  constexpr inline static bool func_v = func<T>::value;
};

/**
 * @brief Partially applies a binary function with first argument fixed
 * @tparam FUNC Binary function to apply
 * @tparam First First argument to fix
 */
template <template <typename, typename> typename FUNC, typename First>
struct binary_partial_apply {
  /**
   * @brief Applies function with first argument fixed
   * @tparam Second Second argument
   */
  template <typename Second>
  using func = FUNC<First, Second>;

  /**
   * @brief Helper alias for func
   * @tparam Second Second argument
   */
  template <typename Second>
  using func_t = typename func<Second>::type;

  /**
   * @brief Helper variable template for func
   * @tparam Second Second argument
   */
  template <typename Second>
  constexpr inline static decltype(func<Second>::value) func_v =
      func<Second>::value;
};

/**
 * @brief Applies multiple metafunctions sequentially
 * @tparam T Initial type
 * @tparam FUNC First metafunction to apply
 * @tparam FUNC_PACK Remaining metafunctions to apply
 */
template <typename T, template <typename> typename FUNC,
          template <typename> typename... FUNC_PACK>
struct sequential_apply
    : sequential_apply<typename FUNC<T>::type, FUNC_PACK...> {};

/**
 * @brief Base case for sequential_apply
 * @tparam T Type to transform
 * @tparam FUNC Metafunction to apply
 */
template <typename T, template <typename> typename FUNC>
struct sequential_apply<T, FUNC> : FUNC<T> {};

/**
 * @brief Helper alias for sequential_apply
 * @tparam T Initial type
 * @tparam FUNC First metafunction to apply
 * @tparam FUNC_PACK Remaining metafunctions to apply
 */
template <typename T, template <typename> typename FUNC,
          template <typename> typename... FUNC_PACK>
using sequential_apply_t =
    typename sequential_apply<T, FUNC, FUNC_PACK...>::type;

/**
 * @brief Helper variable template for sequential_apply
 * @tparam T Initial type
 * @tparam FUNC First metafunction to apply
 * @tparam FUNC_PACK Remaining metafunctions to apply
 */
template <typename T, template <typename> typename FUNC,
          template <typename> typename... FUNC_PACK>
constexpr inline auto sequential_apply_v =
    sequential_apply<T, FUNC, FUNC_PACK...>::value;

/**
 * @brief Checks if a type is a coroutine handle
 * @tparam Type Type to check
 */
template <typename Type>
struct is_coroutine_handle : std::false_type {};

/**
 * @brief Specialization for coroutine handles
 * @tparam Promise Promise type
 */
template <typename Promise>
struct is_coroutine_handle<std::coroutine_handle<Promise>> : std::true_type {};

/**
 * @brief Checks if a type is a valid await_suspend return type
 * @tparam Type Type to check
 */
template <typename Type>
struct is_valid_await_suspend_return_type
    : std::disjunction<std::is_void<Type>, std::is_same<Type, bool>,
                       is_coroutine_handle<Type>> {};

/**
 * @brief Checks if a type has a valid await_suspend method
 * @tparam Type Type to check
 */
template <typename Type>
struct is_valid_await_suspend : std::false_type {};

/**
 * @brief Specialization for coroutine handles
 * @tparam Promise Promise type
 */
template <typename Promise>
struct is_valid_await_suspend_return_type<std::coroutine_handle<Promise>>
    : std::true_type {};

/**
 * @brief Checks if a type has an await_suspend method with valid return type
 * @tparam Type Type to check
 */
template <typename Type>
using is_await_suspend_method = is_valid_await_suspend_return_type<
    decltype(std::declval<Type>().await_suspend(
        std::declval<std::coroutine_handle<>>()))>;

/**
 * @brief Checks if a type has an await_ready method returning bool
 * @tparam Type Type to check
 */
template <typename Type>
using is_await_ready_method =
    std::is_constructible<bool, decltype(std::declval<Type>().await_ready())>;

/**
 * @brief Checks if a type is awaitable
 * @tparam Type Type to check
 */
template <typename Type, typename = std::void_t<>>
struct is_awaitable : std::false_type {};

/**
 * @brief Specialization for awaitable types
 * @tparam Type Awaitable type
 */
template <typename Type>
struct is_awaitable<Type,
                    std::void_t<decltype(std::declval<Type>().await_ready()),
                                decltype(std::declval<Type>().await_suspend(
                                    std::declval<std::coroutine_handle<>>())),
                                decltype(std::declval<Type>().await_resume())>>
    : std::conjunction<is_await_ready_method<Type>,
                       is_await_suspend_method<Type>> {};

/**
 * @brief Helper variable template for is_awaitable
 * @tparam Type Type to check
 */
template <typename Type>
constexpr bool is_awaitable_v = is_awaitable<Type>::value;

/**
 * @brief Checks if a function can be called with given args and returns Ret
 * @tparam Ret Required return type
 * @tparam Fn Function type
 * @tparam Args Argument types
 */
template <typename Ret, typename Fn, typename... Args>
struct is_callable_r : std::false_type {};

/**
 * @brief Specialization for callable functions
 * @tparam Ret Required return type
 * @tparam Fn Function type
 * @tparam Args Argument types
 */
template <typename Ret, typename Fn, typename... Args>
  requires requires(Fn &fn, Args &&...args) {
    { static_cast<Fn &>(fn)(std::forward<Args>(args)...) } -> std::same_as<Ret>;
  }
struct is_callable_r<Ret, Fn, Args...> : std::true_type {};

/**
 * @brief Helper variable template for is_callable_r
 * @tparam Ret Required return type
 * @tparam FN Function type
 * @tparam Args Argument types
 */
template <typename Ret, typename FN, typename... Args>
constexpr inline bool is_callable_r_v = is_callable_r<Ret, FN, Args...>::value;

/**
 * @brief Checks if a type is std::optional
 * @tparam T Type to check
 */
template <class T>
struct is_optional : std::false_type {};

/**
 * @brief Specialization for std::optional
 * @tparam T Optional value type
 */
template <class T>
struct is_optional<std::optional<T>> : std::true_type {};

/**
 * @brief Helper variable template for is_optional
 * @tparam T Type to check
 */
template <class T>
inline constexpr bool is_optional_v = is_optional<T>::value;

/**
 * @brief Alias for std::negation
 * @tparam T Type to negate
 */
template <typename T>
using negation = std::negation<T>;

/**
 * @brief Alias for std::disjunction
 * @tparam T Types to disjunct
 */
template <typename... T>
using disjunction = std::disjunction<T...>;

/**
 * @brief Alias for std::conjunction
 * @tparam T Types to conjunct
 */
template <typename... T>
using conjunction = std::conjunction<T...>;

/**
 * @brief Checks if type T has a converting constructor from U
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
struct is_converting_ctor : std::false_type {};

namespace dyn_optional {
/**
 * @brief Dynamic optional type for internal use
 * @tparam U Value type
 */
template <typename U>
class dynamic_optional {};
}  // namespace dyn_optional

/**
 * @brief Checks if type T can be assigned from U with conversion
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
using is_converting_assign = std::conjunction<
    is_converting_ctor<T, U>,
    std::negation<std::disjunction<
        std::is_assignable<T &, dyn_optional::dynamic_optional<U> &>,
        std::is_assignable<T &, const dyn_optional::dynamic_optional<U> &>,
        std::is_assignable<T &, dyn_optional::dynamic_optional<U> &&>,
        std::is_assignable<T &, const dyn_optional::dynamic_optional<U> &&>>>>;

/**
 * @brief Helper for SFINAE detection
 * @tparam Types Types to ignore
 */
template <typename...>
using try_to_instantiate = void;

/**
 * @brief Dummy type for SFINAE
 */
using disregard_this = void;

/**
 * @brief Implementation of detection idiom
 * @tparam Expression Expression to check
 * @tparam Attempt Helper for SFINAE
 * @tparam Ts Types to check
 */
template <template <typename...> class Expression, typename Attempt,
          typename... Ts>
struct is_detected_impl : std::false_type {
  is_detected_impl(const is_detected_impl &) = delete;
  is_detected_impl(is_detected_impl &&) = delete;
  is_detected_impl &operator=(const is_detected_impl &) = delete;
  is_detected_impl &operator=(is_detected_impl &&) = delete;
};

/**
 * @brief Specialization for detected expressions
 * @tparam Expression Expression that is valid
 * @tparam Ts Types that make expression valid
 */
template <template <typename...> class Expression, typename... Ts>
struct is_detected_impl<Expression, try_to_instantiate<Expression<Ts...>>,
                        Ts...> : std::true_type {};

/**
 * @brief Helper variable template for detection idiom
 * @tparam Expression Expression to check
 * @tparam Ts Types to check with
 */
template <template <typename...> class Expression, typename... Ts>
constexpr bool is_detected =
    is_detected_impl<Expression, disregard_this, Ts...>::value;

/**
 * @brief Expression for assignment operation
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
using assign_expression = decltype(std::declval<T &>() = std::declval<U &>());

/**
 * @brief Helper variable template for assignment check
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
constexpr bool is_assignable = is_detected<assign_expression, T, U>;

/**
 * @brief Checks if type U can be converted to optional-like type T
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
struct is_convertible_to_optional_like {
  static constexpr bool value = !std::disjunction_v<
      std::is_reference<U>, std::is_constructible<T, U &>,
      std::is_constructible<T, const U &>, std::is_constructible<T, U &&>,
      std::is_constructible<T, const U &&>, std::is_convertible<U &, T>,
      std::is_convertible<const U &, T>, std::is_convertible<U &&, T>,
      std::is_convertible<const U &&, T>>;
};

/**
 * @brief Checks if optional-like type T can be assigned from U
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
struct is_assignable_from_optional_like {
  static constexpr bool value =
      is_convertible_to_optional_like<T, U>::value &&
      !std::disjunction_v<
          std::is_assignable<T &, U &>, std::is_assignable<T &, const U &>,
          std::is_assignable<T &, U &&>, std::is_assignable<T &, const U &&>>;
};

/**
 * @brief Checks if a type is optional-like (std::optional or similar)
 * @tparam T Type to check
 */
template <typename T>
struct is_optional_like {
  static constexpr bool value = std::disjunction_v<
      std::is_same<std::optional<std::decay_t<T>>, std::decay_t<T>>,
      std::is_same<std::optional<T>, T>>;
};

/**
 * @brief Checks if type T has a converting constructor from U
 * @tparam T Destination type
 * @tparam U Source type
 */
template <typename T, typename U>
struct is_converting_ctor {
  /**
   * @brief Checks single-argument constructor
   * @tparam U1 Source type
   */
  template <typename U1>
  static constexpr bool value =
      std::is_constructible_v<T, U1> && !std::is_same_v<T, U1> &&
      std::is_convertible_v<U1, T>;

  /**
   * @brief Checks two-argument constructor
   * @tparam U1 First source type
   * @tparam U2 Second source type
   */
  template <typename U1, typename U2>
  static constexpr bool value = std::is_constructible_v<T, U1, U2> &&
                                !(std::is_same_v<T, std::tuple<U1, U2>>) &&
                                std::is_convertible_v<std::tuple<U1, U2>, T>;

  /**
   * @brief Helper for two-argument case
   * @tparam U1 First source type
   * @tparam U2 Second source type
   */
  template <typename U1, typename U2>
  static constexpr bool value_2arg = value<U1, U2>;
};

/**
 * @brief Applies multiple metafunctions sequentially
 * @tparam FUNCS Metafunctions to apply
 */
template <template <typename> typename... FUNCS>
struct sequential_applicator {
  /**
   * @brief Applies metafunctions to type T
   * @tparam T Type to transform
   */
  template <typename T>
  struct func : sequential_apply<T, FUNCS...> {};

  /**
   * @brief Helper alias for func
   * @tparam T Type to transform
   */
  template <typename T>
  using func_t = typename func<T>::type;
};

/**
 * @brief Checks if a type is a specialization of a template
 * @tparam _Type Type to check
 * @tparam _Template Template to check against
 */
template <class _Type, template <class...> class _Template>
constexpr bool is_specialization_v =
    false;  // true if and only if _Type is a specialization of _Template

/**
 * @brief Specialization for template matches
 * @tparam _Template Template to check
 * @tparam _Types Template parameters
 */
template <template <class...> class _Template, class... _Types>
constexpr bool is_specialization_v<_Template<_Types...>, _Template> = true;

/**
 * @brief Type trait for template specialization check
 * @tparam _Type Type to check
 * @tparam _Template Template to check against
 */
template <class _Type, template <class...> class _Template>
struct is_specialization
    : bool_constant<is_specialization_v<_Type, _Template>> {};

/**
 * @brief Checks if type T can be streamed into type S
 *
 * @tparam S The stream type
 * @tparam T The type to be streamed
 * @tparam (unnamed) SFINAE helper parameter
 *
 * Primary template defaults to false_type
 */
template <typename S, typename T, typename = std::void_t<>>
struct is_streamable : std::false_type {};

/**
 * @brief Specialization of is_streamable for streamable types
 *
 * Checks for existence of operator<< between S and T, and ensures S and T are
 * different types
 */
template <typename S, typename T>
struct is_streamable<
    S, T,
    std::void_t<core::meta::meta_funcs::disable_if_t<std::is_same_v<S, T>>,
                decltype(std::declval<std::add_lvalue_reference_t<S>>()
                         << std::declval<T>())>> : std::true_type {};

/// Helper alias for is_streamable::type
template <typename S, typename T>
using is_streamable_t = typeof_t<is_streamable<S, T>>;

/// Helper variable template for is_streamable::value
template <typename S, typename T>
inline constexpr auto is_streamable_v = typev<is_streamable_t<S, T>>;

/**
 * @brief Checks if type T is iterable (has begin() and end())
 *
 * @tparam T Type to check
 * @tparam (unnamed) SFINAE helper parameter
 */
template <typename T, typename = std::void_t<>>
struct is_iterable : std::false_type {};

/// Specialization for iterable types
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

/// Helper variable template for is_iterable::value
template <typename T>
inline constexpr auto is_iterable_v = typev<is_iterable<T>>;

/**
 * @brief Checks if type T is a standard container
 *
 * Verifies presence of container typedefs and methods
 */
template <typename T, typename = std::void_t<>>
struct is_container : std::false_type {};

/// Specialization for container types
template <typename T>
struct is_container<
    T,
    std::void_t<typename T::value_type, typename T::size_type,
                typename T::allocator_type, typename T::iterator,
                typename T::const_iterator, decltype(std::declval<T>().size()),
                decltype(std::declval<T>().begin()),
                decltype(std::declval<T>().end()),
                decltype(std::declval<T>().cbegin()),
                decltype(std::declval<T>().cend())>> : std::true_type {};

/// Helper variable template for is_container::value
template <typename T>
inline constexpr auto is_container_v = typev<is_container<T>>;

/**
 * @brief Checks if type T is a pointer to type U
 *
 * @tparam T Pointer type to check
 * @tparam U Underlying type
 */
template <typename T, typename U>
struct is_pointer_of : std::is_same<T, std::add_pointer_t<U>> {};

/// Helper variable template for is_pointer_of::value
template <typename T, typename U>
inline constexpr auto is_pointer_of_v = typev<is_pointer_of<T, U>>;

/**
 * @brief Checks if type Args is an instantiation of template T
 *
 * @tparam T Template to check against
 * @tparam Args Type to test
 */
template <template <typename...> typename T, typename Args>
struct is_instance_of : std::false_type {};

/// Specialization for matching template instantiations
template <template <typename...> typename T, typename... Args>
struct is_instance_of<T, T<Args...>> : std::true_type {};

/// Helper variable template for is_instance_of::value
template <template <typename...> typename T, typename Args>
inline constexpr auto is_instance_of_v = typev<is_instance_of<T, Args>>;

/**
 * @brief Checks if type Args is a sequence type (like integer_sequence)
 *
 * @tparam T Sequence template to check against
 * @tparam Args Type to test
 */
template <template <typename, auto...> typename T, typename Args>
struct is_sequence_of : std::false_type {};

/// Specialization for sequence types
template <template <typename, auto...> typename T, typename U, auto... values>
struct is_sequence_of<T, T<U, values...>> : std::true_type {};

/// Helper variable template for is_sequence_of::value
template <template <typename, auto...> typename T, typename Args>
inline constexpr auto is_sequence_of_v = typev<is_sequence_of<T, Args>>;

/**
 * @brief Checks if type T is a std::tuple
 */
template <typename T>
struct is_tuple : is_instance_of<std::tuple, T> {};

/// Helper variable template for is_tuple::value
template <typename T>
inline  auto is_tuple_v = typev<is_tuple<T>>;

/**
 * @brief Checks if type T is a sequence (integer_sequence or similar)
 */
template <typename T>
struct is_sequence : is_sequence_of<std::integer_sequence, T> {};

/// Helper variable template for is_sequence::value
template <typename T>
inline constexpr auto is_sequence_v = typev<is_sequence<T>>;

/**
 * @brief Checks if type T is a variadic template with type parameters
 */
template <typename T>
struct is_variadic_type : std::false_type {};

/// Specialization for variadic type templates
template <template <typename...> typename T, typename... Args>
struct is_variadic_type<T<Args...>> : std::true_type {};

/// Helper alias for is_variadic_type::type
template <typename T>
using is_variadic_type_t = typeof_t<is_variadic_type<T>>;

/// Helper variable template for is_variadic_type::value
template <typename T>
inline constexpr auto is_variadic_type_v = typev<is_variadic_type_t<T>>;

/**
 * @brief Checks if type T is a variadic template with value parameters
 */
template <typename T>
struct is_variadic_value : std::false_type {};

/// Specialization for value-only variadic templates
template <template <auto...> typename T, auto... Args>
struct is_variadic_value<T<Args...>> : std::true_type {};

/// Specialization for mixed type/value variadic templates
template <template <typename, auto...> typename T, typename U, auto... Args>
struct is_variadic_value<T<U, Args...>> : std::true_type {};

/// Helper alias for is_variadic_value::type
template <typename T>
using is_variadic_value_t = typeof_t<is_variadic_value<T>>;

/// Helper variable template for is_variadic_value::value
template <typename T>
inline constexpr auto is_variadic_value_v = typev<is_variadic_value_t<T>>;

/**
 * @brief Checks if type T is any kind of variadic template
 */
template <typename T>
struct is_variadic : bool_<is_variadic_type_v<T> || is_variadic_value_v<T>> {};

/// Helper alias for is_variadic::type
template <typename T>
using is_variadic_t = typeof_t<is_variadic<T>>;

/// Helper variable template for is_variadic::value
template <typename T>
inline constexpr auto is_variadic_v = typev<is_variadic_t<T>>;

/**
 * @brief Checks if all types in Args are variadic type templates
 */
template <typename... Args>
struct is_variadic_type_pack : bool_<(is_variadic_type_v<Args> && ...)> {};

/// Helper alias for is_variadic_type_pack::type
template <typename... Args>
using is_variadic_type_pack_t = typeof_t<is_variadic_type_pack<Args...>>;

/// Helper variable template for is_variadic_type_pack::value
template <typename... Args>
inline constexpr auto is_variadic_type_pack_v =
    typev<is_variadic_type_pack_t<Args...>>;

/**
 * @brief Checks if all types in Args are variadic value templates
 */
template <typename... Args>
struct is_variadic_value_pack : bool_<(is_variadic_value_v<Args> && ...)> {};

/// Helper alias for is_variadic_value_pack::type
template <typename... Args>
using is_variadic_value_pack_t = typeof_t<is_variadic_value_pack<Args...>>;

/// Helper variable template for is_variadic_value_pack::value
template <typename... Args>
inline constexpr auto is_variadic_value_pack_v =
    typev<is_variadic_value_pack_t<Args...>>;

/**
 * @brief Checks if all types in Args are any kind of variadic templates
 */
template <typename... Args>
struct is_variadic_pack : bool_<is_variadic_type_pack_v<Args...> ||
                                is_variadic_value_pack_v<Args...>> {};

/// Helper alias for is_variadic_pack::type
template <typename... Args>
using is_variadic_pack_t = typeof_t<is_variadic_pack<Args...>>;

/// Helper variable template for is_variadic_pack::value
template <typename... Args>
inline constexpr auto is_variadic_pack_v = typev<is_variadic_pack_t<Args...>>;

/**
 * @brief Concept checking if Args is a pack of function pointers
 */
template <typename... Args>
concept is_function_pack = requires(Args... args) {
  []<typename... R, typename... T>(R (*...f)(T...)) {}(args...);
};

/**
 * @brief Checks if type T is a group of variadic templates
 */
template <typename T>
struct is_group : std::false_type {};

/// Specialization for group types
template <template <typename...> typename T, typename... Args>
struct is_group<T<Args...>> : is_variadic_pack<Args...> {};

/// Helper variable template for is_group::value
template <typename T>
inline constexpr auto is_group_v = typev<is_group<T>>;

/**
 * @brief Checks if a type is std::optional
 * @tparam T Type to check
 */
template <class T>
struct is_optional : std::false_type {};

/**
 * @brief Specialization for std::optional
 * @tparam T Optional value type
 */
template <class T>
struct is_optional<std::optional<T>> : std::true_type {};

/**
 * @brief Helper variable template for is_optional
 * @tparam T Type to check
 */
template <class T>
inline constexpr bool is_optional_v = is_optional<T>::value;

/**
 * @brief Checks if a type is dynamic_optional
 * @tparam T Type to check
 */
template <class T>
struct is_dynamic_optional : std::false_type {};

/**
 * @brief Specialization for dynamic_optional
 * @tparam T Optional value type
 */
template <class T>
struct is_dynamic_optional<::core::meta::dynamic_optional::dynamic_optional<T>>
    : std::true_type {};

/**
 * @brief Helper variable template for is_dynamic_optional
 * @tparam T Type to check
 */
template <class T>
inline constexpr bool is_dynamic_optional_v = is_dynamic_optional<T>::value;
}  // namespace core::meta::type_traits