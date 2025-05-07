#pragma once

#include "associative_container_concepts.hpp"
namespace core::meta::concepts {
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
}  // namespace core::meta::concepts