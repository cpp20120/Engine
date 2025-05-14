#pragma once
#include <coroutine>
#include <iostream>
#include <string>

namespace core::meta::concepts {

/**
 * @concept has_update
 * @brief Concept to check if a type T has an update method that takes a float
 * and returns void.
 * @tparam T The type to check.
 */
template <typename T>
concept has_update = requires(T t, float deltaTime) {
  { t.update(deltaTime) } -> std::same_as<void>;
};

/**
 * @concept drawable
 * @brief Concept to check if a type T has a draw method that takes a context
 * and returns void.
 * @tparam T The type to check.
 */
template <typename T>
concept drawable = requires(T t, void* context) {
  { t.draw(context) } -> std::same_as<void>;
};

/**
 * @concept serializable
 * @brief Concept to check if a type T has serialize and deserialize methods.
 * @tparam T The type to check.
 */
template <typename T>
concept serializable = requires(T t) {
  { t.serialize() } -> std::same_as<std::string>;
  { t.deserialize(std::declval<std::string>()) } -> std::same_as<void>;
};

/**
 * @concept input_handler
 * @brief Concept to check if a type T has a processInput method that returns
 * void.
 * @tparam T The type to check.
 */
template <typename T>
concept input_handler = requires(T t) {
  { t.processInput() } -> std::same_as<void>;
};

/**
 * @concept printable
 * @brief Concept to check if a type T can be printed to an output stream.
 * @tparam T The type to check.
 */
template <typename T>
concept printable = requires(std::ostream& os, T t) {
  { os << t } -> std::same_as<std::ostream&>;
};

/**
 * @concept is_awaitable
 * @brief Concept to check if a type T is awaitable.
 * @tparam T The type to check.
 */
template <typename T>
concept is_awaitable = requires(T t, std::coroutine_handle<> h) {
  { t.await_ready() } -> std::convertible_to<bool>;
  requires std::same_as<decltype(t.await_suspend(h)), void> ||
               std::same_as<decltype(t.await_suspend(h)), bool> ||
               core::meta::type_traits::is_coroutine_handle<
                   std::remove_cvref_t<decltype(t.await_suspend(h))>>::value;
  t.await_resume();
};

}  // namespace core::meta::concepts
