/**
 * @file sorting_algorithms.hpp
 * @brief Template metaprogramming implementations of various sorting algorithms
 * @ingroup MetaProgramming
 *
 * This header provides compile-time sorting algorithms implemented using
 * template metaprogramming. All algorithms operate on type lists and can be
 * used with custom comparators.
 */

#pragma once

#include "./meta_functions.hpp"

namespace core::meta::algorithms {

/**
 * @brief Bubble sort algorithm implementation for type lists
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * Implements bubble sort by repeatedly swapping adjacent elements if they are
 * in the wrong order. The algorithm runs in O(n²) time complexity at compile
 * time.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct bubble_sort {
  /**
   * @brief Inner sorting implementation
   * @tparam i Current outer loop index
   * @tparam j Current inner loop index
   * @tparam k Fixed upper bound
   * @tparam U Current state of the type list
   */
  template <size_t i, size_t j, size_t k, typename U>
  struct sort
      : sort<i, j - 1, k,
             swap_if_t<predicate_v<comparator, j, j - 1, U>, j, j - 1, U>> {};

  template <size_t i, size_t k, typename U>
  struct sort<i, i, k, U> : sort<i + 1, k, k, U> {};

  template <size_t k, typename U>
  struct sort<k, k, k, U> : std::type_identity<U> {};

  static constexpr auto N = sizeof_t_v<T>;  ///< Length of the type list
  using type =
      typeof_t<sort<0, N - 1, N - 1, T>>;  ///< Resulting sorted type list
};

/// Helper alias for bubble_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using bubble_sort_t = sort_t<bubble_sort, T, comparator>;

/**
 * @brief Shaker sort (bidirectional bubble sort) algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * An improved version of bubble sort that sorts in both directions alternately.
 * More efficient than bubble sort for certain cases.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct shaker_sort {
  /**
   * @brief Next step in the sorting process
   * @tparam i Current index
   * @tparam j Boundary index
   * @tparam k Direction (1 for forward, -1 for backward)
   * @tparam U Current type list state
   * @tparam B Flag indicating if any swaps occurred
   */
  template <int i, int j, int k, typename U, bool B>
  struct next {
    static constexpr auto value = predicate_v<comparator, i + 1, i, U>;
    using cond = swap_if_t<value, i, i + 1, U>;
    using type = typeof_t<next<i + k, j, k, cond, B || value>>;
  };

  template <int j, int k, typename U, bool B>
  struct next<j, j, k, U, B> : std::type_identity<index_type<B, U>> {};

  /**
   * @brief Main sorting implementation
   * @tparam i Left boundary
   * @tparam j Right boundary
   * @tparam U Current type list state
   * @tparam B Flag indicating if sorting should continue
   */
  template <int i, int j, typename U, bool B>
  struct sort {
    using call = typeof_t<next<i, j, 1, U, false>>;

    template <typename V, bool>
    struct impl {
      using curr = typeof_t<next<j - 2, i - 1, -1, V, false>>;
      using type = typeof_t<sort<i + 1, j - 1, typeof_t<curr>, typev<curr>>>;
    };

    template <typename V>
    struct impl<V, false> : std::type_identity<V> {};

    using type = typeof_t<impl<typeof_t<call>, typev<call>>>;
  };

  template <int i, int j, typename U>
  struct sort<i, j, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<0, sizeof_t_v<T> - 1, T, true>>;
};

/// Helper alias for shaker_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using shaker_sort_t = sort_t<shaker_sort, T, comparator>;

/**
 * @brief Odd-even sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * A parallel sorting algorithm that compares odd-indexed elements with their
 * even-indexed neighbors.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct oddeven_sort {
  static constexpr auto N = sizeof_t_v<T>;

  /**
   * @brief Next step in the sorting process
   * @tparam i Current index
   * @tparam U Current type list state
   * @tparam B Flag indicating if any swaps occurred
   * @tparam Condition for continuing
   */
  template <int i, typename U, bool B, bool = i <= N - 2>
  struct next {
    static constexpr auto value = predicate_v<comparator, i + 1, i, U>;
    using cond = swap_if_t<value, i, i + 1, U>;
    using type = typeof_t<next<i + 2, cond, B || value>>;
  };

  template <int i, typename U, bool B>
  struct next<i, U, B, false> : std::type_identity<index_type<B, U>> {};

  /**
   * @brief Main sorting implementation
   * @tparam U Current type list state
   * @tparam B Flag indicating if sorting should continue
   */
  template <typename U, bool B>
  struct sort {
    template <typename V, bool>
    struct impl {
      using call = typeof_t<next<1, V, false>>;
      using curr = typeof_t<next<0, typeof_t<call>, typev<call>>>;
      using type = typeof_t<sort<typeof_t<curr>, typev<curr>>>;
    };

    template <typename V>
    struct impl<V, false> {
      using type =
          typeof_t<type_if<N == 2, next<0, V, false>, index_upper<1, V>>>;
    };

    using type = typeof_t<impl<U, (N > 2)>>;
  };

  template <typename U>
  struct sort<U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<T, true>>;
};

/// Helper alias for oddeven_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using oddeven_sort_t = sort_t<oddeven_sort, T, comparator>;

/**
 * @brief Gnome sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Similar to insertion sort but moving elements to their places by a series of
 * swaps.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct gnome_sort {
  /**
   * @brief Sorting implementation
   * @tparam i Current index
   * @tparam j Upper bound
   * @tparam U Current type list state
   */
  template <int i, int j, typename U>
  struct sort {
    static constexpr auto value = predicate_v<comparator, i - 1, i, U>;
    using next = swap_if_t<!value, i - 1, i, U>;
    using type = typeof_t<sort<value ? i + 1 : i == 1 ? i : i - 1, j, next>>;
  };

  template <int j, typename U>
  struct sort<j, j, U> : std::type_identity<U> {};

  using type = typeof_t<sort<1, sizeof_t_v<T>, T>>;
};

/// Helper alias for gnome_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using gnome_sort_t = sort_t<gnome_sort, T, comparator>;

/**
 * @brief Selection sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * Repeatedly finds the minimum element and puts it at the beginning.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct selection_sort {
  /**
   * @brief Sorting implementation
   * @tparam i Current outer index
   * @tparam j Current inner index
   * @tparam k Upper bound
   * @tparam U Current type list state
   */
  template <size_t i, size_t j, size_t k, typename U>
  struct sort : sort<i, j + 1, k,
                     swap_if_t<predicate_v<comparator, j, i, U>, i, j, U>> {};

  template <size_t i, size_t k, typename U>
  struct sort<i, k, k, U> : sort<i + 1, i + 2, k, U> {};

  template <size_t j, size_t k, typename U>
  struct sort<k, j, k, U> : std::type_identity<U> {};

  using type = typeof_t<sort<0, 1, sizeof_t_v<T>, T>>;
};

/// Helper alias for selection_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using selection_sort_t = sort_t<selection_sort, T, comparator>;

/**
 * @brief Quick sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Divide and conquer algorithm that partitions the list and recursively sorts
 * the partitions.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct quick_sort {
  /**
   * @brief Sorting implementation
   * @tparam p Lower bound
   * @tparam r Upper bound
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <int p, int r, typename U, bool = (p < r)>
  struct sort {
    using pivot = randomized_partition_t<p, r, U, comparator>;
    using left = typeof_t<sort<p, typev<pivot>, typeof_t<pivot>>>;
    using type = typeof_t<sort<typev<pivot> + 1, r, left>>;
  };

  template <int p, int r, typename U>
  struct sort<p, r, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<0, sizeof_t_v<T>, T>>;
};

/// Helper alias for quick_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using quick_sort_t = sort_t<quick_sort, T, comparator>;

/**
 * @brief Iterative quick sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Non-recursive implementation of quick sort using a stack.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct quick_sort_iterative {
  template <bool B, typename U, int p, int r>
  using cond = type_if<B, prepend<U, c_<p>, c_<r>>, std::type_identity<U>>;

  /**
   * @brief Sorting implementation
   * @tparam i Stack index
   * @tparam U Current stack state
   * @tparam V Current type list state
   */
  template <int i, typename U, typename V>
  struct sort {
    static constexpr auto p = get_v<0, U>;
    static constexpr auto r = get_v<1, U>;

    using pivot = randomized_partition_t<p, r, V, comparator>;
    static constexpr int value = typev<pivot>;

    static constexpr auto B1 = value - 1 > p;
    static constexpr auto B2 = value + 1 < r - 1;

    using curr = tail_t<tail_t<U>>;
    using call = cond<B2, cond<B1, curr, p, value>, value + 1, r>;

    using type = typeof_t<sort<i + (B1 + B2 - 1) * 2, call, typeof_t<pivot>>>;
  };

  template <typename U, typename V>
  struct sort<-1, U, V> : std::type_identity<V> {};

  using type =
      typeof_t<sort<1, std::integer_sequence<int, 0, sizeof_t_v<T>>, T>>;
};

/// Helper alias for quick_sort_iterative
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using quick_sort_iterative_t = sort_t<quick_sort_iterative, T, comparator>;

/**
 * @brief Matrix sorting algorithm
 * @tparam T The matrix to sort (flattened)
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Sorts a matrix represented as a flattened type list with indices.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct sort_matrix {
  using indices = matrix_index_sequences_t<T>;
  using outer = first_t<indices>;
  using inner = second_t<indices>;

  /**
   * @brief Sorting implementation
   * @tparam p Lower bound
   * @tparam r Upper bound
   * @tparam U Current matrix state
   * @tparam Condition for continuing
   */
  template <int p, int r, typename U, bool = (p < r)>
  struct sort {
    using pivot = partition_matrix_t<p, r, U, outer, inner, comparator>;
    using left = typeof_t<sort<p, typev<pivot>, typeof_t<pivot>>>;
    using type = typeof_t<sort<typev<pivot> + 1, r, left>>;
  };

  template <int p, int r, typename U>
  struct sort<p, r, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<0, sizeof_t_v<flat_t<T>>, T>>;
};

/// Helper alias for sort_matrix
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using sort_matrix_t = sort_t<sort_matrix, T, comparator>;

/**
 * @brief Stable sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Maintains the relative order of equal elements.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct stable_sort {
  /**
   * @brief Sorting implementation
   * @tparam p Lower bound
   * @tparam r Upper bound
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <int p, int r, typename U, bool = (p < r)>
  struct sort {
    using pivot = randomized_stable_partition_t<p, r, U, comparator>;
    using left = typeof_t<sort<p, typev<pivot> - 1, typeof_t<pivot>>>;
    using type = typeof_t<sort<typev<pivot>, r, left>>;
  };

  template <int p, int r, typename U>
  struct sort<p, r, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<0, sizeof_t_v<T>, T>>;
};

/// Helper alias for stable_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using stable_sort_t = sort_t<stable_sort, T, comparator>;

/**
 * @brief Index-based sorting
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Sorts based on indices while maintaining stability.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct sort_index {
  using type = to_map_t<stable_sort_t<to_map_t<T>, comparator>>;
};

/// Helper alias for sort_index
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using sort_index_t = sort_t<sort_index, T, comparator>;

/**
 * @brief Insertion sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * Builds the final sorted list one item at a time.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct insert_sort {
  /**
   * @brief Sorting implementation
   * @tparam i Current sorted portion size
   * @tparam j Current insertion index
   * @tparam k Upper bound
   * @tparam U Current type list state
   */
  template <int i, int j, int k, typename U>
  struct sort {
    using curr = element_t<j, U>;
    static constexpr auto n = upper_bound_v<curr, U, i, j, comparator>;
    using type = typeof_t<sort<i, j + 1, k, rotate_t<n, j, j + 1, U>>>;
  };

  template <int i, int k, typename U>
  struct sort<i, k, k, U> : std::type_identity<U> {};

  using type = typeof_t<sort<0, 1, sizeof_t_v<T>, T>>;
};

/// Helper alias for insert_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using insert_sort_t = sort_t<insert_sort, T, comparator>;

/**
 * @brief Alternative insertion sort implementation
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct insertion_sort {
  /**
   * @brief Sorting implementation
   * @tparam i Current outer index
   * @tparam j Current inner index
   * @tparam k Upper bound
   * @tparam U Current type list state
   * @tparam V Current element being inserted
   */
  template <size_t i, size_t j, size_t k, typename U, typename V>
  struct sort {
    using prev = element_t<j - 1, U>;

    template <typename W, bool B>
    struct next {
      using type = sort<i, j - 1, k, change_t<j, prev, W>, V>;
    };

    template <typename W>
    struct next<W, false> {
      using cond = element_if_t<i + 1 != k, i + 1, W, std::type_identity<W>>;
      using type = sort<i + 1, i + 1, k, change_t<j, V, W>, cond>;
    };

    using type = typeof_t<typeof_t<next<U, typev<comparator<V, prev>>>>>;
  };

  template <size_t i, size_t k, typename U, typename V>
  struct sort<i, 0, k, U, V> {
    using type = typeof_t<sort<i + 1, i + 1, k, change_t<0, V, U>,
                               element_t<i + 1 == k ? i : i + 1, U>>>;
  };

  template <size_t k, typename U, typename V>
  struct sort<k, k, k, U, V> : std::type_identity<U> {};

  static constexpr auto N = sizeof_t_v<T>;
  using type = typeof_t<sort<1, 1, N, T, element_t<(N > 1), T>>>;
};

/// Helper alias for insertion_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using insertion_sort_t = sort_t<insertion_sort, T, comparator>;

/**
 * @brief Merge helper for advancing through two sorted lists
 * @tparam N Number of elements to merge
 * @tparam T First sorted list
 * @tparam U Second sorted list
 * @tparam comparator Comparison predicate (default: less_t)
 */
template <int N, typename T, typename U,
          template <typename, typename> typename comparator =
              core::meta::meta_funcs::less_t>
  requires core::meta::type_traits::is_variadic_pack_v<T, U>
struct merge_advance {
  /**
   * @brief Merge implementation
   * @tparam i Index in first list
   * @tparam j End of first list
   * @tparam p Index in second list
   * @tparam q End of second list
   * @tparam m Remaining elements to merge
   * @tparam V Result accumulator
   */
  template <int i, int j, int p, int q, int m, typename V>
  struct impl {
    using lhs = element_t<i, T>;
    using rhs = element_t<p, U>;

    static constexpr auto value =
        core::meta::type_traits::typev<comparator<lhs, rhs>>;
    using type = core::meta::type_traits::typeof_t<
        impl<i + value, j, p + !value, q, m - 1,
             append_t<V, std::conditional_t<value, lhs, rhs>>>>;
  };

  template <int i, int j, int p, int q, typename V>
  struct impl<i, j, p, q, 0, V> : std::type_identity<V> {};

  template <int j, int p, int q, int m, typename V>
  struct impl<j, j, p, q, m, V> : concat<V, range_t<p, p + m, U>> {};

  template <int i, int j, int q, int m, typename V>
  struct impl<i, j, q, q, m, V> : concat<V, range_t<i, i + m, T>> {};

  using type = core::meta::type_traits::typeof_t<
      impl<0, sizeof_t_v<T>, 0, sizeof_t_v<U>, N, clear_t<T>>>;
};

/// Helper alias for merge_advance
template <int N, typename T, typename U,
          template <typename, typename> typename comparator =
              core::meta::meta_funcs::less_t>
using merge_advance_t =
    core::meta::type_traits::typeof_t<merge_advance<N, T, U, comparator>>;

/**
 * @brief Merge helper with custom combination logic
 * @tparam F Comparison function
 * @tparam C Combination function
 * @tparam T First sorted list
 * @tparam U Second sorted list
 */
template <template <typename...> typename F, template <typename...> typename C,
          typename T, typename U>
  requires core::meta::type_traits::is_variadic_pack_v<T, U>
struct merge_combine {
  /**
   * @brief Merge implementation
   * @tparam i Index in first list
   * @tparam j End of first list
   * @tparam p Index in second list
   * @tparam q End of second list
   * @tparam V Result accumulator
   */
  template <int i, int j, int p, int q, typename V>
  struct impl {
    using lhs = element_t<i, T>;
    using rhs = element_t<p, U>;

    static constexpr auto value = core::meta::type_traits::typev<F<lhs, rhs>>;
    static constexpr auto B1 = value == 0;
    static constexpr auto B2 = value < 0;
    static constexpr auto B3 = value > 0;

    using next =
        core::meta::type_traits::ternary_conditional_t<!B1, B2, lhs, rhs,
                                                       C<lhs, rhs>>;
    using type = core::meta::type_traits::typeof_t<
        impl<i + B1 + B2, j, p + B1 + B3, q,
             append_t<V, core::meta::type_traits::typeof_t<next>>>>;
  };

  template <int j, int p, int q, typename V>
  struct impl<j, j, p, q, V> : concat<V, range_t<p, q, U>> {};

  template <int i, int j, int q, typename V>
  struct impl<i, j, q, q, V> : concat<V, range_t<i, j, T>> {};

  using type = core::meta::type_traits::typeof_t<
      impl<0, core::meta::meta_funcs::sizeof_t_v<T>, 0,
           core::meta::meta_funcs::sizeof_t_v<U>,
           core::meta::meta_funcs::clear_t<T>>>;
};

/// Helper alias for merge_combine
template <template <typename...> typename F, template <typename...> typename C,
          typename T, typename U>
using merge_combine_t = typeof_t<merge_combine<F, C, T, U>>;

/**
 * @brief Merge operation for merge sort
 * @tparam p Start index
 * @tparam q Middle index
 * @tparam r End index
 * @tparam T Type list to merge
 * @tparam comparator Comparison predicate (default: less_equal_t)
 */
template <size_t p, size_t q, size_t r, typename T,
          template <typename, typename> typename comparator =
              core::meta::meta_funcs::less_equal_t>
struct merge {
  using max = max_element_t<T>;

  template <auto i, auto j, typename U>
  using split = append_t<expand_of_t<U, index_sequence_of_c<i>, j>, max>;

  using left = split<q - p + 1, p, T>;
  using right = split<r - q, q + 1, T>;

  /**
   * @brief Merge implementation
   * @tparam i Index in left half
   * @tparam j Index in right half
   * @tparam k Current output index
   * @tparam l End index
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <size_t i, size_t j, size_t k, size_t l, typename U, bool = k <= l>
  struct assign {
    using first = element_t<i, left>;
    using second = element_t<j, right>;

    static constexpr auto value = typev<comparator<first, second>>;
    using swap = change_t<k, std::conditional_t<value, first, second>, U>;

    using type = typeof_t<assign<i + value, j + !value, k + 1, l, swap>>;
  };

  template <size_t i, size_t j, size_t k, size_t l, typename U>
  struct assign<i, j, k, l, U, false> : std::type_identity<U> {};

  using type = typeof_t<assign<0, 0, p, r, T>>;
};

/// Helper alias for merge
template <size_t p, size_t q, size_t r, typename T,
          template <typename, typename> typename comparator =
              core::meta::meta_funcs::less_equal_t>
using merge_t = typeof_t<merge<p, q, r, T, comparator>>;

/**
 * @brief Merge sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_equal_t)
 *
 * Divide and conquer algorithm that divides the list into halves, sorts them,
 * and merges.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
struct merge_sort {
  /**
   * @brief Sorting implementation
   * @tparam p Start index
   * @tparam r End index
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <size_t p, size_t r, typename U, bool = (p < r)>
  struct sort {
    static constexpr auto q = (p + r) / 2;

    using left = typeof_t<sort<p, q, U>>;
    using right = typeof_t<sort<q + 1, r, left>>;

    using type = merge_t<p, q, r, right, comparator>;
  };

  template <size_t p, size_t r, typename U>
  struct sort<p, r, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<0, sizeof_t_v<T> - 1, T>>;
};

/// Helper alias for merge_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_equal_t>
using merge_sort_t = sort_t<merge_sort, T, comparator>;

/**
 * @brief Strand sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * Works by repeatedly pulling sorted sublists (strands) from the input list and
 * merging them.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct strand_sort {
  using base = clear_t<T>;

  /**
   * @brief Next strand extraction
   * @tparam i Current index
   * @tparam j Strand length
   * @tparam k Upper bound
   * @tparam U Current strand
   * @tparam V Remaining elements
   */
  template <int i, int j, int k, typename U, typename V>
  struct next {
    using curr = element_t<i, V>;
    static constexpr auto value = typev<comparator<back_t<U>, curr>>;

    using lhs = append_if_t<value, U, curr>;
    using rhs = type_if<value, erase_at<i, V>, std::type_identity<V>>;

    using type = typeof_t<next<i + !value, j + 1, k, lhs, rhs>>;
  };

  template <int i, int k, typename U, typename V>
  struct next<i, k, k, U, V> {
    using type = pair_t<U, V>;
  };

  /**
   * @brief Main sorting implementation
   * @tparam U Input list
   * @tparam V Result accumulator
   * @tparam N Size of input list
   * @tparam Condition for continuing
   */
  template <typename U, typename V, auto N = sizeof_t_v<U>, bool = N != 0>
  struct sort {
    static constexpr auto n = sizeof_t_v<V>;

    using call =
        typeof_t<next<0, 0, N - 1, append_t<base, front_t<U>>, tail_t<U>>>;
    using first = first_t<call>;

    using curr = concat_t<V, first>;
    using dest = merge<0, n - 1, sizeof_t_v<curr> - 1, curr, less_equal_t>;

    using cond = type_if<!n, std::type_identity<first>, dest>;
    using type = typeof_t<sort<second_t<call>, cond>>;
  };

  template <typename U, typename V, auto N>
  struct sort<U, V, N, false> : std::type_identity<V> {};

  using type = typeof_t<sort<T, base>>;
};

/// Helper alias for strand_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using strand_sort_t = sort_t<strand_sort, T, comparator>;

/**
 * @brief Heap sort algorithm
 * @tparam T The type list to sort
 * @tparam comparator Comparison predicate (default: less_t)
 *
 * Uses a binary heap to sort elements in ascending order.
 */
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
struct heap_sort {
  static constexpr auto N = sizeof_t_v<T> - 1;

  /**
   * @brief Heapify operation
   * @tparam i Current index
   * @tparam j Heap size
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <int i, int j, typename U, bool = true>
  struct max_heapify {
    static constexpr auto l = 2 * i + 1;
    static constexpr auto r = 2 * (i + 1);

    static constexpr auto k = predicate_v < comparator, i, l <= j ? l : i,
                          U > ? l : i;
    static constexpr auto m = predicate_v < comparator, k, r <= j ? r : k,
                          U > ? r : k;

    using type = typeof_t<max_heapify<m, j, swap_t<i, m, U>, m != i>>;
  };

  template <int i, int j, typename U>
  struct max_heapify<i, j, U, false> : std::type_identity<U> {};

  /**
   * @brief Build max heap
   * @tparam i Current index
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <int i, typename U, bool = (i >= 0)>
  struct build_max_heap
      : build_max_heap<i - 1, typeof_t<max_heapify<i, N, U>>> {};

  template <int i, typename U>
  struct build_max_heap<i, U, false> : std::type_identity<U> {};

  /**
   * @brief Sorting implementation
   * @tparam i Current index
   * @tparam U Current type list state
   * @tparam Condition for continuing
   */
  template <int i, typename U, bool = (i >= 1)>
  struct sort : sort<i - 1, typeof_t<max_heapify<0, i - 1, swap_t<i, 0, U>>>> {
  };

  template <int i, typename U>
  struct sort<i, U, false> : std::type_identity<U> {};

  using type = typeof_t<sort<N, typeof_t<build_max_heap<N / 2, T>>>>;
};

/// Helper alias for heap_sort
template <typename T, template <typename, typename> typename comparator =
                          core::meta::meta_funcs::less_t>
using heap_sort_t = sort_t<heap_sort, T, comparator>;

}  // namespace core::meta::algorithms