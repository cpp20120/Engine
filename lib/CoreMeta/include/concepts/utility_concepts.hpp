#pragma once
#include <iostream>
#include <string>

namespace core::meta::concepts {

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
  { t.serialize() } -> std::same_as<std::string>;
  { t.deserialize(std::declval<std::string>()) } -> std::same_as<void>;
};

template <typename T>
concept input_handler = requires(T t) {
  { t.processInput() } -> std::same_as<void>;
};

template <typename T>
concept printable = requires(std::ostream& os, T t) {
  { os << t } -> std::same_as<std::ostream&>;
};

template <typename T>
concept is_awaitable = requires(Type t, std::coroutine_handle<> h) {
  { t.await_read() } -> std::convertible_to<bool>;
  requires std::same_as<decltype(t.await_suspend(h)), void> ||
               std::same_as<decltype(t.await_suspend(h)), bool> ||
               utils::traits::is_coroutine_handle<
                   std::remove_cvref_t<decltype(t.await_suspend(h))>>::value;
  t.await_resume();
};

}  // namespace core::meta::concepts