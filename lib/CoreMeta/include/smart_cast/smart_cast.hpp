/**
 * @file smart_cast.hpp
 * @brief Advanced polymorphic casting utilities with compile-time safety checks
 * @namespace core::meta::smart_cast
 *
 * This header provides a sophisticated type casting system that extends beyond
 * standard C++ dynamic_cast with additional safety features, compile-time
 * checks, and support for complex type hierarchies and containers.
 */

/**
 * @example smart_cast_example.cpp
 * @brief Example usage of smart_cast functionality
 *
 * This example demonstrates how to use the smart_cast system with a simple
 * class hierarchy.
 *
 * @code{.cpp}
 * // Example class hierarchy
 * struct Base {
 *   virtual ~Base() = default;
 * };
 *
 * struct Derived : Base {
 *   auto cast_to(Base*) -> Base* { return this; }
 * };
 *
 * struct Other : Base {
 *   auto cast_to(Base*) -> Base* { return this; }
 * };
 *
 * // Example usage:
 * Base* base = new Derived();
 *
 * // Basic smart cast
 * auto derived = smart_cast<Derived*>(base);
 * if (derived) {
 *   // Use derived object
 * }
 *
 * // Using with registered casts
 * using CastList = type_list<cast_pair<Derived, Base>, cast_pair<Other, Base>>;
 * auto casted = smart_cast<Derived*>(base, CastList{});
 *
 * // Container support
 * std::vector<Base*> objects = {new Derived(), new Other()};
 * auto derived_objects = smart_cast<Derived*>(objects);
 * @endcode
 *
 * @see smart_cast
 * @see type_list
 * @see cast_pair
 */

#pragma once

#include <array>
#include <concepts>
#include <expected>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <variant>

namespace core::meta::smart_cast {

// --- Concepts for Type Constraints ---

/**
 * @brief Concept checking if a type is polymorphic (has virtual functions)
 * @tparam T Type to check
 */
template <typename T>
concept Polymorphic = std::is_polymorphic_v<std::remove_cvref_t<T>>;

/**
 * @brief Concept checking if a type is a class/struct
 * @tparam T Type to check
 */
template <typename T>
concept ClassType = std::is_class_v<std::remove_cvref_t<T>>;

/**
 * @brief Concept verifying inheritance relationship
 * @tparam Derived Potential derived type
 * @tparam Base Potential base type
 */
template <typename Derived, typename Base>
concept DerivedFrom =
    ClassType<Derived> && ClassType<Base> &&
    std::is_base_of_v<std::remove_cvref_t<Base>, std::remove_cvref_t<Derived>>;

/**
 * @brief Concept validating valid cast pairs
 * @tparam From Source type for casting
 * @tparam To Target type for casting
 */
template <typename From, typename To>
concept ValidCastPair = ClassType<From> && ClassType<To>;

// --- Modern Type Traits ---

namespace traits {

/**
 * @brief Remove pointer qualifiers from type
 * @tparam T Type to transform
 */
template <typename T>
struct remove_pointer {
  using type = T;
};

/// @copydoc remove_pointer
template <typename T>
struct remove_pointer<T*> {
  using type = T;
};

/// @copydoc remove_pointer
template <typename T>
struct remove_pointer<T* const> {
  using type = T;
};

/// @brief Helper alias for remove_pointer
template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

/**
 * @brief Check if type is a pointer
 * @tparam T Type to check
 */
template <typename T>
struct is_pointer {
  static constexpr bool value = false;
};

/// @copydoc is_pointer
template <typename T>
struct is_pointer<T*> {
  static constexpr bool value = true;
};

/// @copydoc is_pointer
template <typename T>
struct is_pointer<T* const> {
  static constexpr bool value = true;
};

/// @brief Helper variable template for is_pointer
template <typename T>
inline constexpr bool is_pointer_v = is_pointer<T>::value;

/**
 * @brief Check if type is a reference
 * @tparam T Type to check
 */
template <typename T>
struct is_reference {
  static constexpr bool value = false;
};

/// @copydoc is_reference
template <typename T>
struct is_reference<T&> {
  static constexpr bool value = true;
};

/// @copydoc is_reference
template <typename T>
struct is_reference<T const&> {
  static constexpr bool value = true;
};

/// @brief Helper variable template for is_reference
template <typename T>
inline constexpr bool is_reference_v = is_reference<T>::value;

/**
 * @brief Check if two types are the same (ignoring cv-qualifiers)
 * @tparam T1 First type
 * @tparam T2 Second type
 */
template <typename T1, typename T2>
struct is_same {
  static constexpr bool value =
      std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>;
};

/// @brief Helper variable template for is_same
template <typename T1, typename T2>
inline constexpr bool is_same_v = is_same<T1, T2>::value;

/**
 * @brief Check if types have base-derived relationship
 * @tparam Derived Potential derived type
 * @tparam Base Potential base type
 */
template <typename Derived, typename Base>
struct is_base_and_derived {
  static constexpr bool value =
      DerivedFrom<Derived, Base> && !is_same_v<Derived, Base>;
};

/// @brief Helper variable template for is_base_and_derived
template <typename Derived, typename Base>
inline constexpr bool is_base_and_derived_v =
    is_base_and_derived<Derived, Base>::value;

// --- STL Container Detection ---

/// @cond
template <typename T, typename = void>
struct has_iterator : std::false_type {};

template <typename T>
struct has_iterator<T, std::void_t<typename T::iterator>> : std::true_type {};

template <typename T, typename = void>
struct has_const_iterator : std::false_type {};

template <typename T>
struct has_const_iterator<T, std::void_t<typename T::const_iterator>>
    : std::true_type {};

template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> : std::true_type {
};

template <typename T, typename = void>
struct has_size_type : std::false_type {};

template <typename T>
struct has_size_type<T, std::void_t<typename T::size_type>> : std::true_type {};
/// @endcond

/**
 * @brief Check if type is an STL-style container
 * @tparam T Type to check
 *
 * Detects containers by checking for required typedefs:
 * - iterator
 * - const_iterator
 * - value_type
 * - size_type
 */
template <typename T>
struct is_stl_container {
  static constexpr bool value =
      has_iterator<T>::value && has_const_iterator<T>::value &&
      has_value_type<T>::value && has_size_type<T>::value;
};

/// @brief Helper variable template for is_stl_container
template <typename T>
inline constexpr bool is_stl_container_v = is_stl_container<T>::value;

}  // namespace traits

// --- Smart Cast Infrastructure ---

/**
 * @brief Metafunction determining cast result type
 * @tparam To Target type
 * @tparam From Source type
 *
 * Preserves pointer-ness of the source type in the result
 */
template <typename To, typename From>
struct cast_result {
  using type = std::conditional_t<traits::is_pointer_v<From>, To*, To>;
};

/// @brief Helper alias for cast_result
template <typename To, typename From>
using cast_result_t = typename cast_result<To, From>::type;

// --- Type List for Cast Registration (Compile-Time) ---

/**
 * @brief Type list for compile-time registration of cast relationships
 * @tparam Types Variadic list of types
 */
template <typename... Types>
struct type_list {};

/**
 * @brief Pair of types representing a cast relationship
 * @tparam From Source type
 * @tparam To Target type
 */
template <typename From, typename To>
struct cast_pair {
  using first_type = From;  ///< Source type
  using second_type = To;   ///< Target type
};

// --- Error Handling with Expected ---

/**
 * @brief Alias for expected type with string error messages
 * @tparam T Expected value type
 * @tparam E Error type (defaults to std::string)
 */
template <typename T, typename E = std::string>
using expected = std::expected<T, E>;

// --- Custom Cast Handlers ---

namespace detail {

/**
 * @brief Handler base for custom cast implementations
 * @tparam To Target type
 * @tparam From Source type
 */
template <typename To, typename From>
struct cast_handler {
  /**
   * @brief Default handler that always fails
   * @param from Source object
   * @return std::nullopt
   */
  static std::optional<To> handle(From from) { return std::nullopt; }
};

/**
 * @brief Cache for dynamic cast results
 * @tparam From Source type
 */
template <typename From>
struct cast_cache {
  std::unordered_map<std::type_index, void*> cache;  ///< Type-to-result mapping

  /**
   * @brief Get cached cast result or perform new cast
   * @tparam To Target type
   * @param from Source object pointer
   * @return To* Cached or newly casted pointer
   */
  template <typename To>
  To* get(From* from) {
    auto it = cache.find(typeid(To));
    if (it != cache.end()) return static_cast<To*>(it->second);
    if (auto result = dynamic_cast<To*>(from)) {
      cache[typeid(To)] = result;
      return result;
    }
    return nullptr;
  }
};

}  // namespace detail

/**
 * @brief Register custom cast handler function
 * @tparam To Target type
 * @tparam From Source type
 * @param handler Function pointer to handler implementation
 */
template <typename To, typename From>
void register_cast_handler(std::optional<To> (*handler)(From)) {
  detail::cast_handler<To, From>::handle = handler;
}

// --- Compile-Time Cast Checking ---

/**
 * @brief Check if types are castable at compile time
 * @tparam To Target type
 * @tparam From Source type
 *
 * Types are castable if:
 * 1. They are the same type (ignoring cv-qualifiers)
 * 2. They have base-derived relationship
 * 3. A custom handler is registered
 */
template <typename To, typename From>
constexpr bool is_castable_v =
    std::is_same_v<std::remove_cvref_t<To>, std::remove_cvref_t<From>> ||
    traits::is_base_and_derived_v<From, To> ||
    requires { detail::cast_handler<To, From>::handle(std::declval<From>()); };

// --- Smart Cast Implementation ---

namespace detail {

/**
 * @brief Core cast implementation with registered pairs
 * @tparam To Target type
 * @tparam From Source type
 * @tparam Pairs Variadic list of registered cast pairs
 * @param list Type list of registered pairs
 * @param from Source object pointer
 * @return std::optional<cast_result_t<To, From>> Cast result or nullopt
 */
template <typename To, typename From, typename... Pairs>
constexpr std::optional<cast_result_t<To, From>> perform_cast(
    type_list<Pairs...>, From* from) noexcept {
  if constexpr (sizeof...(Pairs) == 0) {
    // Base case - no registered pairs
    if constexpr (std::is_same_v<std::remove_cvref_t<To>,
                                 std::remove_cvref_t<From>>) {
      return from;  // Same type
    } else if constexpr (traits::is_base_and_derived_v<From, To>) {
      return static_cast<To*>(from);  // Upcast
    }
    return detail::cast_handler<To, From>::handle(from);  // Custom handler
  } else {
    // Recursive case - check registered pairs
    bool found = false;
    std::optional<cast_result_t<To, From>> result;
    [&]<size_t... I>(std::index_sequence<I...>) {
      auto try_cast = [&]<size_t Index>(std::integral_constant<size_t, Index>) {
        using Pair = std::tuple_element_t<Index, std::tuple<Pairs...>>;
        if constexpr (traits::is_same_v<typename Pair::first_type, From> &&
                      traits::is_same_v<typename Pair::second_type, To>) {
          found = true;
          if constexpr (requires {
                          std::declval<From>()->cast_to(std::declval<To*>());
                        }) {
            result = from->cast_to(static_cast<To*>(nullptr));  // Member cast
          } else if constexpr (DerivedFrom<To, From>) {
            result = static_cast<To*>(from);  // Downcast
          }
        }
      };
      (try_cast(std::integral_constant<size_t, I>{}), ...);
    }(std::index_sequence_for<Pairs...>{});

    if (found) return result;
    return perform_cast<To, From>(type_list<>{}, from);  // Recurse
  }
}

// --- Container Support ---

/**
 * @brief Cast elements within an STL container
 * @tparam To Target element type
 * @tparam From Container type
 * @tparam TypeList Type list of registered pairs
 * @param container Input container
 * @param list Type list
 * @return Container with casted elements
 */
template <typename To, typename From, typename TypeList>
  requires traits::is_stl_container_v<From>
constexpr auto smart_cast_container(From&& container, TypeList list) {
  using value_type = typename std::remove_cvref_t<From>::value_type;
  using result_type = cast_result_t<To, value_type>;
  using container_type = std::remove_cvref_t<From>;

  container_type result;
  for (auto&& elem : container) {
    if (auto casted =
            smart_cast<To>(std::forward<decltype(elem)>(elem), list)) {
      if constexpr (traits::is_pointer_v<value_type>) {
        if (*casted) result.insert(result.end(), *casted);  // Filter nulls
      } else {
        result.insert(result.end(), *casted);
      }
    }
  }
  return result;
}

// --- Tuple Support ---

/**
 * @brief Cast elements within a tuple
 * @tparam To Target element type
 * @tparam FromTypes Tuple element types
 * @tparam TypeList Type list of registered pairs
 * @tparam Is Index sequence
 * @param tuple Input tuple
 * @param list Type list
 * @param seq Index sequence
 * @return Tuple with casted elements
 */
template <typename To, typename... FromTypes, typename TypeList, size_t... Is>
constexpr auto smart_cast_tuple_impl(const std::tuple<FromTypes...>& tuple,
                                     TypeList list,
                                     std::index_sequence<Is...>) {
  return std::make_tuple(smart_cast<To>(std::get<Is>(tuple), list)...);
}

// --- Variant Support ---

/**
 * @brief Cast variant alternative
 * @tparam To Target type
 * @tparam FromTypes Variant alternative types
 * @tparam TypeList Type list of registered pairs
 * @param variant Input variant
 * @param list Type list
 * @return std::optional<To> Cast result or nullopt
 */
template <typename To, typename... FromTypes, typename TypeList>
constexpr std::optional<To> smart_cast_variant(
    const std::variant<FromTypes...>& variant, TypeList list) {
  return std::visit(
      [&](auto&& arg) -> std::optional<To> {
        return smart_cast<To>(std::forward<decltype(arg)>(arg), list);
      },
      variant);
}

// --- Cached Casting ---

/**
 * @brief Cached cast implementation
 * @tparam To Target type
 * @tparam From Source type
 * @param from Source object
 * @return std::optional<To> Cached result or nullopt
 */
template <typename To, typename From>
constexpr std::optional<To> cached_smart_cast_impl(From from) {
  thread_local detail::cast_cache<std::remove_pointer_t<From>> cache;
  if (auto result = cache.template get<To>(from)) {
    return result;
  }
  return std::nullopt;
}

// --- Multi-level Casting ---

/**
 * @brief Cast through intermediate types
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs
 * @param from Source object
 * @param list Type list
 * @return std::optional<To> Final cast result or nullopt
 */
template <typename To, typename From, typename TypeList>
constexpr auto multi_level_smart_cast_impl(From from, TypeList list) {
  if constexpr (is_castable_v<To, From>) {
    return smart_cast<To>(from, list);  // Direct cast
  } else {
    using Intermediate = std::remove_pointer_t<From>;
    if (auto intermediate = smart_cast<Intermediate>(from, list)) {
      return multi_level_smart_cast_impl<To>(*intermediate, list);  // Recurse
    }
    return std::optional<To>{};
  }
}

// --- Debug Casting ---

/**
 * @brief Log debug message
 * @param message Message to log
 */
inline void debug_log(const char* message) {
  std::cerr << "[smart_cast] " << message << std::endl;
}

/**
 * @brief Debug cast with logging
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs
 * @param from Source object
 * @param list Type list
 * @param location Call location string
 * @return std::optional<To> Cast result or nullopt
 */
template <typename To, typename From, typename TypeList>
constexpr std::optional<To> debug_smart_cast_impl(From from, TypeList list,
                                                  const char* location = "") {
  if (auto result = smart_cast<To>(from, list)) {
    return result;
  }
  debug_log((std::string("Failed cast from ") + typeid(From).name() + " to " +
             typeid(To).name() + " at " + location)
                .c_str());
  return std::nullopt;
}

}  // namespace detail

// --- Main Smart Cast Interface ---

/**
 * @brief Primary smart cast function
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs (default empty)
 * @param from Source object
 * @param list Type list (default empty)
 * @return Cast result or empty optional
 *
 * Supports:
 * - Polymorphic types
 * - STL containers
 * - Tuples
 * - Variants
 */
template <typename To, typename From, typename TypeList = type_list<>>
  requires(Polymorphic<From> &&
           (traits::is_pointer_v<From> || traits::is_reference_v<From>)) ||
          traits::is_stl_container_v<From> ||
          requires { std::tuple_size<std::remove_cvref_t<From>>::value; } ||
          requires { std::variant_size<std::remove_cvref_t<From>>::value; }
constexpr auto smart_cast(From from, TypeList list = {}) {
  using CleanFrom = traits::remove_pointer_t<std::remove_cvref_t<From>>;

  if constexpr (traits::is_stl_container_v<From>) {
    return detail::smart_cast_container<To>(std::forward<From>(from), list);
  } else if constexpr (requires {
                         std::tuple_size<std::remove_cvref_t<From>>::value;
                       }) {
    return detail::smart_cast_tuple_impl<To>(
        from, list,
        std::make_index_sequence<
            std::tuple_size_v<std::remove_cvref_t<From>>>{});
  } else if constexpr (requires {
                         std::variant_size<std::remove_cvref_t<From>>::value;
                       }) {
    return detail::smart_cast_variant<To>(from, list);
  } else {
    if (!from) {
      return std::optional<cast_result_t<To, CleanFrom>>{};
    }
    return detail::perform_cast<To, CleanFrom>(list, from);
  }
}

// --- Extended Interfaces ---

/**
 * @brief Smart cast with rich error information
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs (default empty)
 * @param from Source object
 * @param list Type list (default empty)
 * @return expected<cast_result_t<To, From>, std::string>
 *         Cast result or error message
 */
template <typename To, typename From, typename TypeList = type_list<>>
constexpr expected<cast_result_t<To, From>, std::string> smart_cast_ex(
    From from, TypeList list = {}) {
  if (auto result = smart_cast<To>(from, list)) {
    return *result;
  }
  return std::unexpected("Cast failed");
}

/**
 * @brief Cached version of smart_cast
 * @tparam To Target type
 * @tparam From Source type
 * @param from Source object
 * @return std::optional<To> Cached result or nullopt
 */
template <typename To, typename From>
constexpr std::optional<To> cached_smart_cast(From from) {
  return detail::cached_smart_cast_impl<To>(from);
}

/**
 * @brief Multi-level smart cast
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs (default empty)
 * @param from Source object
 * @param list Type list (default empty)
 * @return Cast result through intermediate types or empty optional
 */
template <typename To, typename From, typename TypeList = type_list<>>
constexpr auto multi_level_smart_cast(From from, TypeList list = {}) {
  return detail::multi_level_smart_cast_impl<To>(from, list);
}

/**
 * @brief Debug version of smart_cast with logging
 * @tparam To Target type
 * @tparam From Source type
 * @tparam TypeList Type list of registered pairs (default empty)
 * @param from Source object
 * @param list Type list (default empty)
 * @param location Call location string (default empty)
 * @return std::optional<To> Cast result or nullopt
 */
template <typename To, typename From, typename TypeList = type_list<>>
constexpr std::optional<To> debug_smart_cast(From from, TypeList list = {},
                                             const char* location = "") {
  return detail::debug_smart_cast_impl<To>(from, list, location);
}

// --- Macro for Declaring Cast Specializations ---

/**
 * @def DECLARE_CAST_SPECIALIZATION
 * @brief Declare a cast specialization using a member method
 * @param To Target type
 * @param From Source type
 * @param Method Member method name to call for casting
 */
#define DECLARE_CAST_SPECIALIZATION(To, From, Method)                  \
  namespace core::meta::smart_cast {                                   \
  template <>                                                          \
  inline std::optional<cast_result_t<To, From>>                        \
  smart_cast<To, From, type_list<>>(From from, type_list<>) noexcept { \
    if (!from) return std::nullopt;                                    \
    return from->Method();                                             \
  }                                                                    \
  }

/**
 * @def DECLARE_BIDIRECTIONAL_CAST
 * @brief Declare bidirectional cast specializations
 * @param From First type
 * @param To Second type
 * @param FromMethod Method for From->To cast
 * @param ToMethod Method for To->From cast
 */
#define DECLARE_BIDIRECTIONAL_CAST(From, To, FromMethod, ToMethod) \
  DECLARE_CAST_SPECIALIZATION(To, From, FromMethod)                \
  DECLARE_CAST_SPECIALIZATION(From, To, ToMethod)

}  // namespace core::meta::smart_cast