#pragma once
#include <functional>
#include <type_traits>

namespace core::meta::concepts {

template <typename F, typename... Args>
concept InvocableWith =
    requires(F f, Args... args) { std::invoke(f, args...); };

template <typename F>
concept Curryable = requires(F f) {
  {
    std::invoke(f,
                std::declval<std::decay_t<decltype(std::placeholders::_1)>>())
  } -> std::same_as<std::invoke_result_t<F, decltype(std::placeholders::_1)>>;
};

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



}  // namespace core::meta::concepts