#pragma once

#include <iostream>
#include <type_traits>
#include <utility>

namespace core::meta::type_signature {

#if defined(__clang__) || defined(__GNUC__)
#define FUNC_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define FUNC_SIGNATURE __FUNCSIG__
#else
#define FUNC_SIGNATURE "Unknown compiler"
#endif

template <typename Ret, typename... Args>
struct FunctionSignature {
  using ReturnType = std::type_identity_t<Ret>;
  using ArgumentTypes = std::tuple<std::type_identity_t<Args>...>;

  static void print(const char* name) {
    std::cout << name << " :: " << signature() << '\n';
  }

 private:
  static std::string signature() {
    return type_list<Args...>() + " -> " + type_name<Ret>();
  }

  template <typename T>
  static std::string type_name() {
    return typeid(std::type_identity_t<T>).name();
  }

  template <typename... Ts>
  static std::string type_list() {
    return ((type_name<Ts>() + " -> ") + ...);
  }
};

#define PRINT_FUNCTION_SIGNATURE(func) \
  FunctionSignature<decltype(func)>::print(#func)
}  // namespace core::meta::type_signature