/**
 * @file reflection.hpp
 * @brief Core meta-programming reflection for C++ types
 * @ingroup core_meta
 */
#pragma once

#include <any>
#include <array>
#include <iterator>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
namespace core::meta::reflection {
/**
 * @brief A view adapter that presents a range in reverse order
 * @tparam Range The range type to reverse
 */
template <typename Range>
class ReverseView {
  const Range& range;

 public:
  /**
   * @brief Constructs a ReverseView for the given range
   * @param r The range to reverse
   */
  explicit ReverseView(const Range& r) : range(r) {}

  /**
   * @brief Iterator for ReverseView
   */
  class Iterator {
    using BaseIterator = decltype(std::rbegin(std::declval<Range>()));
    BaseIterator iter;

   public:
    /**
     * @brief Constructs an iterator
     * @param it The base iterator to reverse
     */
    explicit Iterator(BaseIterator it) : iter(it) {}

    /**
     * @brief Dereference operator
     * @return Reference to the current element
     */
    auto operator*() const { return *iter; }

    /**
     * @brief Prefix increment
     * @return Reference to this iterator
     */
    Iterator& operator++() {
      --iter;
      return *this;
    }

    /**
     * @brief Postfix increment
     * @return Copy of iterator before increment
     */
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    /**
     * @brief Equality comparison
     * @param other Iterator to compare with
     * @return True if iterators are equal
     */
    bool operator==(const Iterator& other) const { return iter == other.iter; }

    /**
     * @brief Inequality comparison
     * @param other Iterator to compare with
     * @return True if iterators are not equal
     */
    bool operator!=(const Iterator& other) const { return iter != other.iter; }
  };

  /**
   * @brief Returns iterator to beginning
   * @return Reverse iterator to beginning
   */
  Iterator begin() { return Iterator(std::rbegin(range)); }

  /**
   * @brief Returns iterator to end
   * @return Reverse iterator to end
   */
  Iterator end() { return Iterator(std::rend(range)); }

  /**
   * @brief Returns const iterator to beginning
   * @return Const reverse iterator to beginning
   */
  Iterator cbegin() const { return Iterator(std::rbegin(range)); }

  /**
   * @brief Returns const iterator to end
   * @return Const reverse iterator to end
   */
  Iterator cend() const { return Iterator(std::rend(range)); }
};

/**
 * @brief Creates a ReverseView for the given range
 * @tparam Range The range type
 * @param range The range to reverse
 * @return ReverseView of the range
 */
template <typename Range>
ReverseView<Range> Reverse(const Range& range) {
  return ReverseView<Range>(range);
}

/**
 * @brief Cache for reflection data of type T
 * @tparam T The type to cache reflection data for
 */
template <typename T>
struct ReflectionCache {
  static constexpr auto Data =
      ComputeReflection<T>();  ///< Cached reflection data
};

/**
 * @brief Retrieves cached reflection data for type T
 * @tparam T The type to get reflection data for
 * @return Constexpr reference to cached reflection data
 */
template <typename T>
constexpr auto& GetCachedReflection() {
  return ReflectionCache<T>::Data;
}

/**
 * @brief Wrapper for type attributes
 * @tparam T The attribute type
 */
template <typename T>
struct Attribute {
  /**
   * @brief Constructs an Attribute with given value
   * @param value The attribute value
   */
  constexpr Attribute(T value) : Value(value) {}
  T Value;  ///< The attribute value
};

/**
 * @brief Checks if a field has a specific attribute
 * @tparam Attr The attribute type to check for
 * @tparam Field The field type
 * @param field The field to check
 * @return True if the field has the attribute, false otherwise
 */
template <typename Attr, typename Field>
constexpr bool HasAttribute(const Field& field) {
  return field.template HasAttr<Attr>();
}

/**
 * @brief Gets an attribute from a field
 * @tparam Attr The attribute type to get
 * @tparam Field The field type
 * @param field The field to get the attribute from
 * @return The attribute value
 */
template <typename Attr, typename Field>
constexpr auto GetAttribute(const Field& field) {
  return field.template GetAttr<Attr>();
}

/**
 * @brief Gets constructor argument types as a tuple
 * @tparam T The type whose constructor to analyze
 * @tparam ...Args Constructor argument types
 * @param ... Unused parameter
 * @return Tuple containing constructor argument types
 * @internal
 */
template <typename T, typename... Args>
constexpr auto ConstructorInfo(T (*)(Args...)) {
  return std::tuple<Args...>{};
}

/**
 * @brief Gets constructor argument types for type T
 * @tparam T The type to analyze
 * @return Tuple containing constructor argument types
 */
template <typename T>
constexpr auto GetConstructorArgs() {
  return ConstructorInfo(static_cast<T* (*)()>(nullptr));
}

/**
 * @brief Gets the type name as a string view
 * @tparam T The type to get name for
 * @return String view containing type name
 * @internal
 */
template <typename T>
constexpr auto GetTypeName() {
  constexpr std::string_view Prefix = "auto __cdecl GetTypeName<";
  constexpr std::string_view Suffix = ">(void)";
  constexpr std::string_view Function = "__FUNCSIG__";

  return Function.substr(Prefix.size(),
                         Function.size() - Prefix.size() - Suffix.size());
}

/**
 * @brief Gets description of a pointer member
 * @tparam Ptr Pointer to member
 * @return String view containing description
 * @internal
 */
template <auto Ptr>
constexpr auto GetDescription() {
  constexpr std::string_view prefix = "auto __cdecl getDescription<";
  constexpr std::string_view suffix = ">(void)";
  constexpr std::string_view func = "__FUNCSIG__";

  constexpr auto start = func.find(prefix) + prefix.size();
  constexpr auto end = func.rfind(suffix);
  constexpr std::string_view full = func.substr(start, end - start);

  constexpr auto pos = full.rfind("::");
  return full.substr(pos + 2);
}

/**
 * @brief Type list for template metaprogramming
 * @tparam ...Args Types in the list
 * @internal
 */
template <typename... Args>
struct TypeList {};

/**
 * @brief Gets template arguments as TypeList
 * @tparam T Template type
 * @tparam ...Args Template arguments
 * @param t Template instance
 * @return TypeList containing template arguments
 * @internal
 */
template <template <typename...> class T, typename... Args>
constexpr auto GetTemplateArgs(T<Args...>) {
  return TypeList<Args...>{};
}

/**
 * @brief Wrapper for compile-time values
 * @tparam I The wrapped value
 * @internal
 */
template <auto I>
struct Wrapper {};

/**
 * @brief Type list implementation
 * @tparam ...Ts Types in the list
 * @internal
 */
template <typename... Ts>
struct TypeList {
  using Types = std::tuple<Ts...>;                    ///< Tuple representation
  static constexpr std::size_t Size = sizeof...(Ts);  ///< Number of types
};

/**
 * @brief Gets first type from TypeList
 * @tparam T First type
 * @tparam ...Ts Remaining types
 * @return First type
 * @internal
 */
template <typename T, typename... Ts>
constexpr auto Head(TypeList<T, Ts...>) -> T {
  return {};
}

/**
 * @brief Gets tail of TypeList (all but first)
 * @tparam T First type
 * @tparam ...Ts Remaining types
 * @return TypeList of remaining types
 * @internal
 */
template <typename T, typename... Ts>
constexpr auto Tail(TypeList<T, Ts...>) -> TypeList<Ts...> {
  return {};
}

/**
 * @brief Checks if TypeList contains type T
 * @tparam T Type to check for
 * @tparam List TypeList to check
 * @internal
 */
template <typename T, typename List>
constexpr bool Contains = false;

/**
 * @brief Specialization for Contains check
 * @tparam T Type to check for
 * @tparam ...Ts Types in list
 * @internal
 */
template <typename T, typename... Ts>
constexpr bool Contains<T, TypeList<Ts...>> = (std::is_same_v<T, Ts> || ...);

/**
 * @brief Compares two TypeLists for equality
 * @tparam ...Ts First list types
 * @tparam ...TTs Second list types
 * @return True if lists are equal
 * @internal
 */
template <typename... Ts, typename... TTs>
consteval auto operator==(const TypeList<Ts...>&, const TypeList<TTs...>&)
    -> bool {
  return false;
}

/**
 * @brief Specialization for equal TypeLists
 * @tparam ...Ts Types in both lists
 * @return True since lists are equal
 * @internal
 */
template <typename... Ts>
consteval auto operator==(const TypeList<Ts...>&, const TypeList<Ts...>&)
    -> bool {
  return true;
}

/**
 * @brief Helper for type casting in reflection
 * @tparam I Index
 * @tparam T Type
 * @internal
 */
template <std::size_t I, typename T>
struct IndexedType {};

/**
 * @brief Helper for type casting in reflection
 * @tparam ... Variadic parameters
 * @internal
 */
template <typename...>
struct Caster {};

/**
 * @brief Specialization for type casting helper
 * @tparam ...Is Indices
 * @tparam ...Ts Types
 * @internal
 */
template <std::size_t... Is, typename... Ts>
struct Caster<std::index_sequence<Is...>, Ts...> : IndexedType<Is, Ts>... {};

/**
 * @brief Gets type at index I from TypeList
 * @tparam I Index to get
 * @tparam ...Ts Types in list
 * @return Type at index I
 * @internal
 */
template <std::size_t I, typename... Ts>
constexpr auto Get(TypeList<Ts...>)
    -> decltype([]<typename T>(Impl::IndexedType<I, T>&&) -> T {
    }(Impl::Caster<std::index_sequence_for<Ts...>, Ts...>{}));

/**
 * @brief Helper for compile-time value injection
 * @tparam I Index
 * @internal
 */
template <auto I>
struct Getter {
  friend constexpr auto GetId(Getter<I>);
};

/**
 * @brief Helper for compile-time value injection
 * @tparam I Index
 * @tparam Value Value to inject
 * @internal
 */
template <auto I, auto Value>
struct Injector {
  friend constexpr auto GetId(Getter<I>) { return Value; };
};

// Static assertions for compile-time testing
static_assert((std::ignore = Injector<0, 42>(), true));
static_assert(GetId(Getter<0>{}) == 42);

/**
 * @brief Helper for type casting in reflection
 * @tparam Main Main type
 * @tparam I Index
 * @internal
 */
template <typename Main, auto I>
struct Caster {
  template <typename T,
            auto = Injector<TypeList<Main, Wrapper<I>>{}, TypeList<T>{}>{}>
  constexpr operator T&&();

  template <typename T,
            auto = Injector<TypeList<Main, Wrapper<I>>{}, TypeList<T>{}>{}>
  constexpr operator T&();
};

/**
 * @brief Simplified type caster
 * @internal
 */
struct SimpleCaster {
  template <typename T>
  constexpr operator T&&();

  template <typename T>
  constexpr operator T&();
};

/**
 * @brief Implementation helper for getting argument count
 * @tparam T Type to analyze
 * @tparam Max Maximum number to try
 * @tparam ...Is Indices
 * @return Number of constructor arguments
 * @internal
 */
template <typename T, std::size_t Max, std::size_t... Is>
consteval auto GetArgsCountImpl() {
  if constexpr (requires { T{(Is, SimpleCaster{})...}; }) {
    return sizeof...(Is);
  } else {
    static_assert(sizeof...(Is) != Max, "Constructor not found");
    return GetArgsCountImpl<T, Is..., 0>();
  }
}

/**
 * @brief Gets number of constructor arguments for type T
 * @tparam T Type to analyze
 * @tparam Max Maximum number to try (default 256)
 * @return Number of constructor arguments
 */
template <typename T, std::size_t Max = 256>
consteval auto GetArgsCount() {
  return GetArgsCountImpl<T, Max, 0, 0>();
}

/**
 * @brief Reflects type T to get its member types
 * @tparam T Type to reflect
 * @tparam I Number of members (defaults to argument count)
 * @return TypeList of member types
 */
template <typename T, std::size_t I = GetArgsCount<T>()>
consteval auto Reflect() {
  return [&]<auto... Is>(std::index_sequence<Is...>)
    requires requires { T{Caster<T, Is>{}...}; }
  {
    return TypeList<typename decltype(GetId(
        Getter<TypeList<T, Wrapper<Is>>{}>{}))::Type...>{};
  }(std::make_index_sequence<I>());
}

/**
 * @brief Tag with value helper
 * @tparam Tag Tag type
 * @tparam Value Associated value
 * @internal
 */
template <typename Tag, auto Value>
struct TagWithValue {};

/**
 * @brief Counter implementation helper
 * @tparam Tag Tag type
 * @tparam I Current index
 * @tparam ...Ts Types to count
 * @return Count of types
 * @internal
 */
template <typename Tag, std::size_t I = 0, typename... Ts>
consteval auto CounterImpl() -> std::size_t {
  if constexpr (requires { GetId(Getter<TagWithValue<Tag, I>{}>{}); }) {
    return CounterImpl<Tag, I + 1, Ts...>();
  }
  return (std::ignore = Injector<TagWithValue<Tag, I>{}, 0>{}, I);
}

/**
 * @brief Compile-time counter for types
 * @tparam Tag Tag type (default void)
 * @tparam ...Ts Types to count
 * @tparam R Result from CounterImpl
 * @return Count of types
 * @internal
 */
template <typename Tag = void, typename... Ts,
          auto R = CounterImpl<Tag, 0, Ts...>()>
constexpr auto Counter() -> std::size_t {
  return R;
}

/**
 * @brief Helper for unique key generation
 * @tparam Tag Tag type
 * @tparam Index Key index
 * @internal
 */
template <typename Tag, std::size_t Index>
struct GetUniqueKey {};

/**
 * @brief Gets unique types from TypeList
 * @tparam ...Ts Types in list
 * @return TypeList of unique types
 * @internal
 */
template <typename... Ts>
consteval auto GetUnique(TypeList<Ts...>) {
  (
      [] {
        constexpr auto I = Counter<TypeList<Ts...>, Ts>();
        std::ignore =
            Injector<GetUniqueKey<TypeList<Ts...>, I>{}, TypeList<Ts>{}>{};
      }(),
      ...);
  return []<std::size_t... Is>(std::index_sequence<Is...>) {
    return TypeList<typename decltype(GetId(
        Getter<GetUniqueKey<TypeList<Ts...>, Is>{}>{}))...>{};
  }(std::make_index_sequence<Counter<TypeList<Ts...>>()>());
}

/**
 * @brief Key for meta info storage
 * @tparam Current Current type
 * @tparam I Index
 * @internal
 */
template <typename Current, std::size_t I>
struct InfoKey {};

/**
 * @brief Injector for meta info
 * @tparam Current Current type
 * @internal
 */
template <typename Current>
struct InfoInjector {
  template <typename T, typename... Args,
            std::size_t I = Counter<Current, TypeList<T, Args...>>(),
            auto = Injector<InfoKey<Current, I>{}, TypeList<T, Args...>{}>{}>
  static auto Method(Args...) -> void;
};

/**
 * @brief Injects type info for reflection
 * @tparam T Type to inject
 * @internal
 */
template <typename T>
consteval auto Inject() {
  []<typename... Ts>(TypeList<Ts...>) {}(GetUnique(KTypeList<T>));
}

/**
 * @brief Gets types from method info
 * @tparam T Type to analyze
 * @tparam ...Ts Additional types
 * @return TypeList of types
 * @internal
 */
template <typename T, typename... Ts>
consteval auto GetTFromMethod() {
  return []<std::size_t... Is>(std::index_sequence<Is...>) {
    return TypeList<decltype(Get<0>(GetId(Getter<InfoKey<T, Is>{}>{})))...>{};
  }(std::make_index_sequence<Counter<T>()>());
}

/**
 * @brief Drops first type from TypeList
 * @tparam T First type
 * @tparam ...Ts Remaining types
 * @return TypeList without first type
 * @internal
 */
template <typename T, typename... Ts>
constexpr auto DropHead(TypeList<T, Ts...>) -> TypeList<Ts...> {
  return {};
}

/**
 * @brief Gets argument types from method info
 * @tparam T Type to analyze
 * @return TypeList of argument types
 * @internal
 */
template <typename T>
consteval auto GetArgsFromMethod() {
  return []<std::size_t... Is>(std::index_sequence<Is...>) {
    return TypeList<decltype(DropHead(GetId(Getter<InfoKey<T, Is>{}>{})))...>{};
  }(std::make_index_sequence<Counter<T>()>());
}

/**
 * @brief Meta information storage
 * @tparam T Type being described
 * @tparam Types Additional types
 * @internal
 */
template <typename T, typename Types>
struct MetaInfo {
  static constexpr std::size_t KTypeId = Counter<Types, T>();  ///< Type ID
  using Type = T;  ///< The described type

 private:
  static constexpr auto _ = Injector<MetaInfoKey<KTypeId>{}, TypeList<T>{}>{};
};

/**
 * @brief Gets type ID for T
 * @tparam T Type to get ID for
 */
template <typename T>
inline constexpr std::size_t KTypeId = MetaInfo<T>::KTypeId;

/**
 * @brief Gets meta info by ID
 * @tparam Id Type ID
 * @internal
 */
template <std::size_t Id>
using GetMetaInfo = MetaInfo<typename decltype(utils::reflection::GetId(
    Getter<MetaInfoKey<Id>{}>{}))::Type>;

/**
 * @brief Gets type by ID
 * @tparam Id Type ID
 * @internal
 */
template <std::size_t Id>
using GetType = GetMetaInfo<Id>::Type;

// Example usage of Reverse with array
constexpr auto Array = std::array{KTypeId<int>, KTypeId<void>} | Reverse;

/**
 * @brief Checks if Derived inherits from Base
 * @tparam Derived Derived type
 * @tparam Base Base type
 * @return True if Derived inherits from Base
 */
template <typename Derived, typename Base>
constexpr bool IsBaseOf() {
  return std::is_base_of_v<Base, Derived>;
}

/**
 * @brief Helper for base type extraction
 * @tparam T Type to analyze
 * @tparam Void SFINAE helper
 * @internal
 */
template <typename T, typename = void>
struct BaseTypesHelper {
  using Type = TypeList<>;  ///< Empty type list by default
};

/**
 * @brief Specialization when T has BaseTypes
 * @tparam T Type with BaseTypes
 * @internal
 */
template <typename T>
struct BaseTypesHelper<T, std::void_t<typename T::BaseTypes>> {
  using Type = typename T::BaseTypes;  ///< Inherited BaseTypes
};

/**
 * @brief Gets base types for type T
 * @tparam T Type to analyze
 * @return TypeList of base types
 */
template <typename T>
constexpr auto GetBaseTypes() {
  return typename BaseTypesHelper<T>::Type{};
};

/**
 * @namespace core::meta::reflection::Describe
 * @brief Namespace containing type description utilities
 */
namespace Describe {
/**
 * @brief Empty attribute list
 * @tparam Ts Attribute types
 */
template <typename... Ts>
struct Attrs {
  using GetAttrs = Attrs;
};
/**
 * @brief Tag type for compile-time type identification
 * @tparam T The tagged type
 */
template <typename T>
struct Tag {
  using Type = T;
};
/**
 * @brief Field descriptor for reflection
 * @tparam Field_t The field pointer or enum value
 */
template <auto Field_t>
struct Field;
/**
 * @brief Gets the reflection description for type T
 * @tparam T The type to describe
 * @return Description object containing reflection data
 */
template <typename T>
constexpr auto Get();

namespace Detail {

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
auto GetMemPtrType(T) -> Tag<T>;

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
auto GetMemPtrClass(T) -> Tag<T>;

template <class C, typename T>
auto GetMemPtrType(T C::*) -> Tag<T>;

template <class C, typename T>
auto GetMemPtrClass(T C::*) -> Tag<C>;

template <class C, typename T, typename... Args>
auto GetMemPtrType(T (C::*v)(Args...)) -> Tag<decltype(v)>;

template <class C, typename T, typename... Args>
auto GetMemPtrClass(T (C::*)(Args...)) -> Tag<C>;

template <bool Methods, typename... T>
constexpr size_t Count() {
  return ((1 * (Methods == T::IsMethod)) + ... + 0);
}

constexpr std::string_view NextName(std::string_view& src) {
  auto resStart = src.find("::");
  resStart = resStart == std::string_view::npos ? src.find_first_not_of(" ")
                                                : resStart + 2;
  auto resEnd = src.find_first_of(" \t\n\r,", resStart);
  auto result = src.substr(resStart, resEnd - resStart);
  src = src.substr(src.find_first_of(',', resEnd) + 1);
  return result;
}

template <typename T, typename... Ts>
constexpr bool Has(Describe::Attrs<Ts...>) {
  return (false || ... || std::is_base_of_v<T, Ts>);
}

template <typename T>
auto GetAttr(Describe::Attrs<>) -> Describe::Tag<void> {
  return {};
}

template <typename T, typename Head, typename... Ts>
auto GetAttrs(Describe::Attrs<Head, Ts...>) {
  if constexpr (std::is_base_of_v<T, Head>) {
    return Tag<Head>{};
  } else {
    return GetAttr<T>(Attrs<Ts...>{});
  }
}

template <typename T, typename = void>
struct HasAttrs : std::false_type {};

template <typename T>
struct HasAttrs<
    T, std::void_t<decltype(Describe::Detail::GetAttrs(Describe::Tag<T>{}))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasFieldAttrs : std::false_type {};

template <typename T>
struct HasFieldAttrs<T, std::void_t<decltype(Describe::Detail::GetAttrs(
                            Describe::Tag<typename T::Cls>{}, T{}))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasStaticAttrs : std::false_type {};

template <typename T>
struct HasStaticAttrs<T, std::void_t<typename T::GetAttrs>> : std::true_type {};

template <size_t Idx, typename Head, typename... Ts>
struct PackIdx : PackIdx<Idx - 1, Ts...> {};

template <typename Head, typename... Ts>
struct PackIdx<0, Head, Ts...> {
  using Type = Head;
};
}  // namespace Detail
/**
 * @brief Field descriptor for reflection
 * @tparam Field_t The field pointer or enum value
 */
template <auto _Field>
struct Field {
  static_assert(std::is_member_pointer_v<decltype(Field)> ||
                    std::is_enum_v<decltype(Field)>,
                "Field can only be used with &_::members or enums");
  std::string_view Name;
  static constexpr auto Value = Field;
  using Type = typename decltype(Describe::Detail::GetMemPtrType(Field))::Type;
  static constexpr auto IsMethod = std::is_member_function_pointer_v<Type>;
  static constexpr auto IsEnum = std::is_enum_v<Type>;
  using Cls = typename decltype(Describe::Detail::GetMemPtrClass(Field))::Type;

  template <typename T>
  static constexpr auto&& Get(T&& obj) noexcept {
    return std::forward<T>(obj).*Field;
  }

  template <typename T, typename... Args>
  constexpr auto Invoke(T&& obj, Args&&... args) const {
    if constexpr (IsMethod) {
      return (std::forward<T>(obj).*Field)(std::forward<Args>(args)...);
    } else {
      static_assert(IsMethod, "Field is not a method");
    }
  }

  static constexpr bool IsStatic =
      std::is_member_pointer_v<decltype(Field)> &&
      std::is_same_v<decltype(Field), std::nullptr_t>;
};
/**
 * @brief Type description containing field and method information
 * @tparam Cls The described class type
 * @tparam Parent The parent class type (for inheritance)
 * @tparam Fields Field descriptors
 */
template <typename Cls, typename Parent, typename... Fields>
struct Description : protected Fields... {
  using Type = Cls;       ///< The described type
  using Parent = Parent;  ///< Parent type (void if none)
  std::string_view Name;  ///< Name of the type
  static constexpr auto IsEnum = std::is_enum_v<Cls>;  ///< True if enum
  static constexpr auto FieldsCount =
      Describe::Detail::Count<false, Fields...>();  ///< Number of fields
  static constexpr auto MethodsCount =
      Describe::Detail::Count<true, Fields...>();  ///< Number of methods
  static constexpr auto AllCount =
      FieldsCount + MethodsCount;  ///< Number of fields and methods
  static constexpr auto NPos = size_t(-1);
  /**
   * @brief Gets field by member pointer
   * @tparam Member Member pointer
   * @return Reference to field descriptor
   */
  template <auto Member>
  constexpr auto& Get() {
    return static_cast<Field<Member>&>(*this);
  }

  template <auto Member>
  constexpr const auto& Get() const {
    return static_cast<const Field<Member>&>(*this);
  }
  /**
   * @brief Iterates over all fields
   * @tparam F Callable type
   * @param f Callable to invoke for each field
   */
  template <typename F>
  constexpr void ForEachField(F&& f) const {
    auto helper = [&f](auto field) {
      if constexpr (field.IsMethod == false) f(field);
    };
    (helper(static_cast<const Fields&>(*this)), ...);
  }
  /**
   * @brief Iterates over all methods
   * @tparam F Callable type
   * @param f Callable to invoke for each method
   */
  template <typename F>
  constexpr void ForEachMethod(F&& f) const {
    auto helper = [&f](auto field) {
      if constexpr (field.IsMethod == true) f(field);
    };
    (helper(static_cast<const Fields&>(*this)), ...);
  }
  /**
   * @brief Iterates over all members (fields and methods)
   * @tparam F Callable type
   * @param f Callable to invoke for each member
   */
  template <typename F>
  constexpr void ForEach(F&& f) const {
    (static_cast<void>(f(static_cast<const Fields&>(*this))), ...);
  }

  template <auto F>
  static constexpr size_t IndexOf(Field<F>) {
    size_t result = NPos;
    size_t count = size_t(-1);
    ((count++, std::is_same_v<Field<F>, Fields> && (result = count)), ...);
    return result;
  }

  template <auto F>
  static constexpr size_t IndexOf() {
    return IndexOf<F>(Field<F>{});
  }

  constexpr size_t IndexOf(std::string_view name) const {
    size_t result = NPos;
    size_t count = size_t(-1);
    ((count++,
      (static_cast<const Fields&>(*this).Name == name) && (result = count)),
     ...);
    return result;
  }
};

template <size_t Idx, typename Cls, typename Parent, typename... Fields>
constexpr auto& ByIndex(Description<Cls, Parent, Fields...>& desc) {
  using F = typename Detail::PackIdx<Idx, Fields...>::Type;
  return desc.template Get<F::Value>();
}

template <size_t Idx, typename Cls, typename Parent, typename... Fields>
constexpr const auto& ByIndex(Description<Cls, Parent, Fields...> const& desc) {
  using F = typename PackIdx<Idx, Fields...>::Type;
  return desc.template Get<F::Value>();
}

template <typename F, typename = void>
struct GetAttrs {
  using Type = Attrs<>;
};

template <typename C, typename = void>
struct HasAttrs : std::false_type {};

template <typename C>
struct GetAttrs<C, std::enable_if_t<HasAttrs<C>::value>> {
  using Type = typename C::GetAttrs;
};

template <typename C>
struct GetAttrs<C, std::enable_if_t<HasAttrs<C>::value>> {
  using Type = decltype(GetAttrs(Tag<C>{}));
};

template <auto _Field>
struct GetAttrs<Field<_Field>,
                std::enable_if_t<Detail::HasFieldAttrs<Field<_Field>>::value>> {
  using Type =
      decltype(GetAttrs(Tag<typename Field<Field>::Cls>{}, Field<Field>{}));
};

template <typename Cls, typename... Rest>
struct GetAttrs<Description<Cls, Rest...>> : GetAttrs<Cls> {};

template <typename Any>
struct GetAttrs<const Any> : GetAttrs<Any> {};

template <typename T>
using GetAttrsT = typename GetAttrs<T>::Type;

template <typename T, typename From>
using ExtractAttrT =
    typename decltype(Detail::GetAttr<T>(GetAttrsT<From>{}))::Type;

template <typename T, typename Who>
constexpr bool HasAttrV = Detail::Has<T>(GetAttrsT<Who>{});

template <typename T, typename = void>
struct IsDescribed : std::false_type {};

template <typename T>
struct IsDescribed<T, std::void_t<decltype(GetDescription(Tag<T>{}))>>
    : std::true_type {};
/**
 * @brief Checks if type T has reflection data
 * @tparam T Type to check
 */
template <typename T>
constexpr auto IsDescribedV = IsDescribed<T>::value;
/**
 * @brief Checks if type T is a described struct
 * @tparam T Type to check
 */
template <typename T>
constexpr auto IsDescribedStructV = IsDescribed<T>::value && !std::is_enum_v<T>;
/**
 * @brief Checks if type T is a described enum
 * @tparam T Type to check
 */
template <typename T>
constexpr auto IsDescribedEnumV = IsDescribed<T>::value && std::is_enum_v<T>;

template <typename T>
constexpr auto Get() {
  static_assert(IsDescribedV<T>, "Please use DESCRIBE() macro");
  constexpr auto res = GetDescription(Tag<T>{});
  return res;
}

template <typename Cls, auto... Fields>
constexpr auto Describe(std::string_view clsName, std::string_view names) {
  Description<Cls, void, Field<Fields>...> result = {};
  result.Name = clsName;
  (static_cast<void>(result.template Get<Fields>().Name =
                         Detail::NextName(names)),
   ...);
  return result;
}

template <typename Cls, auto... Fields, typename ParCls, typename ParCls2,
          auto... ParFields>
constexpr auto Describe(
    Description<ParCls, ParCls2, Field<ParFields>...> parent,
    std::string_view clsName, std::string_view names) {
  Description<Cls, ParCls, Field<ParFields>..., Field<Fields>...> result = {};
  result.Name = clsName;
  (static_cast<void>(result.template Get<ParFields>().Name =
                         parent.template Get<ParFields>()),
   ...);
  (static_cast<void>(result.template Get<Fields>().Name =
                         Detail::NextName(names)),
   ...);
  return result;
}

template <typename Cls>
struct _ {
  template <auto... Fs, typename... Args>
  static constexpr auto Desc(Args... args) {
    return Describe<Cls, Fs...>(args...);
  }
};
/**
 * @brief Description specialization for enum types
 * @tparam E The enum type
 * @tparam Values Enum values and names
 */
template <typename E>
struct EnumDescription {
  std::string_view Name;
  std::array<std::pair<E, std::string_view>, sizeof...(Values)> Values;
};
/**
 * @brief Serializes an object to string using reflection
 * @tparam T Object type
 * @param obj Object to serialize
 * @return String representation
 */
template <typename T>
std::string Serialize(const T& obj) {
  std::string result;
  auto desc = GetDescription(Tag<T>{});
  desc.ForEachField([&](auto field) {
    result += field.Name + ": " + std::to_string(field.Get(obj)) + "\n";
  });
  return result;
}
/**
 * @brief Creates an instance of type T using reflection
 * @tparam T Type to instantiate
 * @return New instance
 */
template <typename T>
T CreateInstance() {
  auto desc = GetDescription(Tag<T>{});
  return T{desc.template Get<0>().Create(), desc.template Get<1>().Create()};
};

template <typename T>
struct TypeDescription;

template <typename T>
struct TypeDescription<std::vector<T>> {
  static constexpr std::string_view Name = "std::vector";
  using ValueType = T;
};
template <template <typename...> typename Template, typename... Ts>
struct TypeDescription<Template<Ts...>> {
  static constexpr auto Name = "TemplateInstance";
};

template <typename T, typename = void>
struct has_reflect : std::false_type {};

template <typename T>
struct has_reflect<T, std::void_t<decltype(T::reflect())>> : std::true_type {};

template <typename T>
concept Reflectable = has_reflect<T>::value;

constexpr uint32_t fnv1a_hash(const char* str) {
  constexpr uint32_t fnv1a_prime = 16777619u;
  constexpr uint32_t fnv1a_offset_basis = 2166136261u;

  uint32_t hash = fnv1a_offset_basis;
  while (*str) {
    hash ^= static_cast<uint32_t>(*str++);
    hash *= fnv1a_prime;
  }
  return hash;
}

template <typename T>
struct FieldHash {
  const char* name;
  uint32_t hash;
  size_t offset;

  constexpr FieldHash(const char* name, size_t offset)
      : name(name), hash(fnv1a_hash(name)), offset(offset) {};
};

template <typename T>
constexpr auto get_fields() {
  return std::array{
      FieldHash<T>{"field1", offsetof(T, field1)},
      FieldHash<T>{"field2", offsetof(T, field2)},
      FieldHash<T>{"field3", offsetof(T, field3)},
  };
}

template <typename T>
constexpr FieldHash<T> get_field_by_name(const char* name) {
  uint32_t hash = fnv1a_hash(name);
  for (const auto& field : get_fields<T>()) {
    if (field.hash == hash) {
      return field;
    }
  }
  throw std::runtime_error("Field not found");
}

template <typename Base, typename Derived>
constexpr auto get_all_fields() {
  auto base_fields = get_fields<Base>();
  auto derived_fields = get_fields<Derived>();

  std::array merged_fields = base_fields;
  merged_fields.insert(merged_fields.end(), derived_fields.begin(),
                       derived_fields.end());

  return merged_fields;
}

template <typename T>
bool IsType(const std::any& obj) {
  return obj.type() == typeid(T);
}

template <typename... Ts>
using UniqueTypes = std::variant<std::decay_t<Ts>...>;

using FieldValue = decltype([] {
  using VariantType = typename UniqueTypes<
      std::decay_t<decltype(std::declval<MyClass>().intField)>,
      std::decay_t<decltype(std::declval<MyClass>().doubleField)>,
      std::decay_t<decltype(std::declval<MyClass>().stringField)>,
      std::decay_t<decltype(std::declval<MyClass>().boolField)>>::type;
  return VariantType{};
})();

struct FieldValueContainer {
  std::string_view name;
  FieldValue value;
};

template <typename T>
constexpr auto GetFields() {
  return std::array{REGISTER_FIELD(T, intField), REGISTER_FIELD(T, doubleField),
                    REGISTER_FIELD(T, stringField),
                    REGISTER_FIELD(T, boolField)};
}

template <typename T>
std::vector<FieldValueContainer> GetFieldValues(const T& obj) {
  std::vector<FieldValueContainer> result;
  constexpr auto fields = GetFields<T>();

  for (const auto& field : fields) {
    result.push_back({field.name, std::invoke(field.ptr, obj)});
  }

  return result;
}

template <typename T, typename... Attributes>
struct Field : public Attributes... {
  using Type = T;
  constexpr Field(std::string_view name) : Name(name) {}
  std::string_view Name;
};

template <typename Cls, typename Parent, typename... Fields>
struct Description : protected Fields... {
  using Type = Cls;
  using ParentType = Parent;
  std::string_view Name;
  static constexpr auto IsEnum = std::is_enum_v<Cls>;
  static constexpr auto FieldsCount = sizeof...(Fields);

  template <auto Member>
  constexpr auto& Get() {
    return static_cast<Field<Member>&>(*this);
  }

  template <typename F>
  constexpr void ForEachField(F&& f) const {
    auto helper = [&f](auto& field) {
      if constexpr (!field.IsMethod) f(field);
    };
    (helper(static_cast<const Fields&>(*this)), ...);
  }

  template <typename F>
  constexpr void ForEachMethod(F&& f) const {
    auto helper = [&f](auto& field) {
      if constexpr (field.IsMethod) f(field);
    };
    (helper(static_cast<const Fields&>(*this)), ...);
  }

  template <typename F>
  constexpr void ForEach(F&& f) const {
    (static_cast<void>(f(static_cast<const Fields&>(*this))), ...);
  }

  constexpr void PrintTree(int indent = 0) const {
    auto indent_str = std::string(indent, ' ');
    std::cout << indent_str << Name << "\n";
    ForEachField([&](const auto& field) {
      std::cout << indent_str << "  " << field.Name << "\n";
    });
    ForEachMethod([&](const auto& method) {
      std::cout << indent_str << "  " << method.Name << "()\n";
    });
  }
};

template <typename T, template <typename...> class Template>
struct is_instantiation_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_instantiation_of<Template<Args...>, Template> : std::true_type {};

template <typename T, template <typename...> class Template>
inline constexpr bool is_instantiation_of_v =
    is_instantiation_of<T, Template>::value;

template <typename T, template <typename..., auto...> class Template>
struct is_instantiation_of_mixed : std::false_type {};

template <template <typename...> class Template, typename... Types,
          auto... Values>
struct is_instantiation_of_mixed<Template<Types..., Values...>, Template>
    : std::true_type {};

template <typename T, template <typename..., auto...> class Template>
inline constexpr bool is_instantiation_of_mixed_v =
    is_instantiation_of_mixed<T, Template>::value;

template <typename T, template <typename..., auto...> class Template>
struct ExtractTemplateParameters;

template <template <typename...> class Template, typename... Types,
          auto... Values>
struct ExtractTemplateParameters<Template<Types..., Values...>, Template> {
  using type =
      std::tuple<Types..., std::integral_constant<decltype(Values), Values>...>;
};

template <typename T, template <typename..., auto...> class Template>
using ExtractTemplateParameters_t =
    typename ExtractTemplateParameters<T, Template>::type;

// GetTemplateParameters для смешанных шаблонов
template <typename T, template <typename..., auto...> class Template>
constexpr auto GetTemplateParameters() {
  static_assert(is_instantiation_of_mixed_v<T, Template>,
                "T должен быть специализацией указанного смешанного шаблона");
  return ExtractTemplateParameters_t<T, Template>{};
}
#define REGISTER_FIELD(Struct, Field) \
  FieldInfo<Struct, decltype(Struct::Field)> { #Field, &Struct::Field }
/**
 * @brief Macro to declare an enum for reflection
 * @param EnumType Enum type name
 * @param ... Enum values
 */
#define DESCRIBE_ENUM(EnumType, ...)                                         \
  inline constexpr auto GetDescription(::utils::Reflection::Tag<EnumType>) { \
    constexpr std::array values = {__VA_ARGS__};                             \
    return EnumDescription<EnumType>{#EnumType, values};                     \
  }
/**
 * @brief Macro to allow description of a class
 * @param cls Class name
 */
#define ALLOW_DESCRIBE_FOR(cls) \
  friend constexpr auto GetDescription(::utils::Reflection::Tag<cls>);

#define _D_DESCRIBE(cls, ...) \
  ::utils::Reflection::_<cls>::template Desc<__VA_ARGS__>
/**
 * @brief Macro to describe a class
 * @param cls Class name
 * @param ... Member pointers
 */
#define DESCRIBE(cls, ...)                                              \
  inline constexpr auto GetDescription(::utils::Reflection::Tag<cls>) { \
    using _ [[maybe_unused]] = cls;                                     \
    return _D_DESCRIBE(cls, __VA_ARGS__)(#cls, #__VA_ARGS__);           \
  }
/**
 * @brief Macro to describe an inherited class
 * @param cls Class name
 * @param parent Parent class name
 * @param ... Member pointers
 */
#define DESCRIBE_INHERIT(cls, parent, ...)                                   \
  inline constexpr auto GetDescription(::utils::Reflection::Tag<cls>) {      \
    using _ = cls;                                                           \
    return _D_DESCRIBE(cls, __VA_ARGS__)(::utils::Reflection::Get<parent>(), \
                                         #cls, #__VA_ARGS__);                \
  }

#define DESCRIBE_TEMPL_CLASS(...)              \
  inline constexpr auto GetDescription(        \
      ::utils::Reflection::Tag<__VA_ARGS__>) { \
    using _ = __VA_ARGS__;                     \
    constexpr std::string_view _clsName = #__VA_ARGS__;

#define DESCRIBE_TEMPL_FIELDS(...)                            \
  return _D_DESCRIBE(_, __VA_ARGS__)(_clsName, #__VA_ARGS__); \
  }
/**
 * @brief Macro to describe attributes for a type
 * @param cls Class name
 * @param ... Attribute types
 */
#define DESCRIBE_ATTRS(cls, ...)                                  \
  inline constexpr auto GetAttrs(::utils::Reflection::Tag<cls>) { \
    return ::utils::Reflection::Attrs<__VA_ARGS__>{};             \
  }

#define DESCRIBE_GET(cls, ...)                                    \
  inline constexpr auto GetAttrs(::utils::Reflection::Tag<cls>) { \
    return ::utils::Reflection::Attrs<__VA_ARGS__>{};             \
  }

}  // namespace Describe
}  // namespace core::meta::reflection