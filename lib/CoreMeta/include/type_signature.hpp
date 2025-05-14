#pragma once

#include <iostream>
#include <type_traits>
#include <utility>

namespace core::meta::type_signature {

/**
 * @def FUNC_SIGNATURE
 * @brief Macro to get the function signature as a string.
 *
 * This macro is defined differently based on the compiler being used.
 * It uses __PRETTY_FUNCTION__ for Clang and GCC, __FUNCSIG__ for MSVC,
 * and defaults to "Unknown compiler" for other compilers.
 */
#if defined(__clang__) || defined(__GNUC__)
#define FUNC_SIGNATURE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define FUNC_SIGNATURE __FUNCSIG__
#else
#define FUNC_SIGNATURE "Unknown compiler"
#endif

/**
 * @class FunctionSignature
 * @brief Template class representing the signature of a function.
 *
 * This class captures the return type and argument types of a function
 * and provides a method to print the function signature.
 *
 * @tparam Ret The return type of the function.
 * @tparam Args The argument types of the function.
 */
template <typename Ret, typename... Args>
struct FunctionSignature {
  /**
   * @typedef ReturnType
   * @brief Type alias for the return type of the function.
   */
  using ReturnType = std::type_identity_t<Ret>;

  /**
   * @typedef ArgumentTypes
   * @brief Type alias for the tuple of argument types of the function.
   */
  using ArgumentTypes = std::tuple<std::type_identity_t<Args>...>;

  /**
   * @brief Prints the function signature.
   *
   * This static method prints the name and signature of the function.
   *
   * @param name The name of the function.
   */
  static void print(const char* name) {
    std::cout << name << " :: " << signature() << '\n';
  }

 private:
  /**
   * @brief Generates the function signature as a string.
   *
   * This static method generates a string representation of the function
   * signature.
   *
   * @return A string representing the function signature.
   */
  static std::string signature() {
    return type_list<Args...>() + " -> " + type_name<Ret>();
  }

  /**
   * @brief Gets the name of a type as a string.
   *
   * This static method uses typeid to get the name of a type.
   *
   * @tparam T The type to get the name of.
   * @return A string representing the type name.
   */
  template <typename T>
  static std::string type_name() {
    return typeid(std::type_identity_t<T>).name();
  }

  /**
   * @brief Generates a list of type names as a string.
   *
   * This static method generates a string representation of a list of type
   * names.
   *
   * @tparam Ts The types to get the names of.
   * @return A string representing the list of type names.
   */
  template <typename... Ts>
  static std::string type_list() {
    return ((type_name<Ts>() + " -> ") + ...);
  }
};

/**
 * @def PRINT_FUNCTION_SIGNATURE(func)
 * @brief Macro to print the function signature.
 *
 * This macro uses the FunctionSignature class to print the signature of a
 * function.
 *
 * @param func The function to print the signature of.
 */
#define PRINT_FUNCTION_SIGNATURE(func) \
  FunctionSignature<decltype(func)>::print(#func)
}  // namespace core::meta::type_signature
