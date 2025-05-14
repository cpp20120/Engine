#pragma once
#include <functional>
#include <initializer_list>
#include <iterator>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>

#include "./concepts/aggregate_concepts.hpp"
#include "./concepts/associative_container_concepts.hpp"
#include "./concepts/basic_concepts.hpp"
#include "./concepts/comparasion_concepts.hpp"
#include "./concepts/container_concepts.hpp"
#include "./concepts/container_of_concepts.hpp"
#include "./concepts/converting_concepts.hpp"
#include "./concepts/function_concepts.hpp"
#include "./concepts/iterator_concepts.hpp"
#include "./concepts/map_container_concepts.hpp"
#include "./concepts/optional_concepts.hpp"
#include "./concepts/utility_concepts.hpp"

#include "./traits/type_traits.hpp"

namespace core::meta::concepts {

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


}  // namespace core::meta::concepts