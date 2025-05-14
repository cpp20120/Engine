#pragma once
#include <functional>
#include <type_traits>

#include "../traits/type_traits.hpp"

/**
 * @namespace core::meta::concepts
 * @brief Namespace for meta-programming concepts and utilities.
 */
namespace core::meta::concepts {

/**
 * @concept InvocableWith
 * @brief Concept to check if a callable F can be invoked with arguments Args.
 * @tparam F The callable type to check.
 * @tparam Args The argument types to check against.
 */
template <typename F, typename... Args>
concept InvocableWith =
    requires(F f, Args... args) { std::invoke(f, args...); };

/**
 * @concept Curryable
 * @brief Concept to check if a callable F is curryable.
 * @tparam F The callable type to check.
 */
template <typename F>
concept Curryable = requires(F f) {
  {
    std::invoke(f,
                std::declval<std::decay_t<decltype(std::placeholders::_1)>>())
  } -> std::same_as<std::invoke_result_t<F, decltype(std::placeholders::_1)>>;
};

/**
 * @concept invocable_as
 * @brief Concept to check if a callable Fn is invocable with the given
 * signatures.
 * @tparam Fn The callable type to check.
 * @tparam Signatures The function signatures to check against.
 */
template <typename Fn, typename... Signatures>
concept invocable_as = requires(Signatures&... signatures) {
  ([]<typename Ret, typename... Args>(auto (&)(Args...)->Ret)
     requires std::is_invocable_r_v<Ret, Fn, Args...>
   {}(signatures),
   ...);
};

/**
 * @concept callable_as
 * @brief Concept to check if a callable Fn is callable with the given
 * signatures.
 * @tparam Fn The callable type to check.
 * @tparam Signatures The function signatures to check against.
 */
template <typename Fn, typename... Signatures>
concept callable_as = requires(Signatures&... signatures) {
  ([]<typename Ret, typename... Args>(auto (&)(Args...)->Ret)
     requires core::meta::type_traits::is_callable_r_v<Ret, Fn, Args...>
   {}(signatures),
   ...);
};

/**
 * @concept hash_function
 * @brief Concept to check if a callable Fn is a hash function for KeyType.
 * @tparam Fn The callable type to check.
 * @tparam KeyType The key type to check against.
 */
template <typename Fn, typename KeyType>
concept hash_function = callable_as<Fn const, auto(KeyType&)->std::size_t,
                                    auto(KeyType const&)->std::size_t>;

}  // namespace core::meta::concepts
