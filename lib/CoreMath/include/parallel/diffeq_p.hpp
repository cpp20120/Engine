#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>

namespace core::math::diffeq::parallel {
/**
 * @brief A utility class for executing parallel operations using Intel TBB.
 *
 * This class provides a wrapper around Intel Threading Building Blocks (TBB)
 * to perform parallel for-loops and invoke multiple functions concurrently.
 * It's especially useful for computational tasks such as solving differential
 * equations where performance is critical.
 *
 * The template parameter is reserved for future use to enable compile-time
 * selection of execution policies (e.g., parallel or serial execution),
 * replacing the runtime `parallelize_` flag.
 *
 * @tparam T Reserved for future use as a parallelization policy tag.
 */

template <typename T>
class ParallelExecutor {
  [[maybe_unused]] T reserved_type_marker;

 public:
  explicit ParallelExecutor(
      [[maybe_unused]] size_t max_concurrency = tbb::task_arena::automatic)
      : arena_(max_concurrency) {}
  /**
   * @brief Executes a parallel for-loop over a range of indices.
   *
   * Runs the provided function `f` in parallel for each index in the range
   * [start, end).
   *
   * @tparam Func Callable type that accepts a single `size_t` index.
   * @param start Starting index (inclusive).
   * @param end Ending index (exclusive).
   * @param f Function to execute for each index.
   */
  template <typename Func>
  void parallel_for(size_t start, size_t end, Func f) {
    arena_.execute([&] {
      tbb::parallel_for(tbb::blocked_range<size_t>(start, end),
                        [&](const tbb::blocked_range<size_t>& range) {
                          for (size_t i = range.begin(); i != range.end();
                               ++i) {
                            f(i);
                          }
                        });
    });
  }
  /**
   * @brief Constructs the executor with an optional concurrency limit.
   *
   * @param max_concurrency Maximum number of threads to use. Defaults to
   * automatic detection.
   */
  template <typename Func>
  void parallel_for_each(auto&& container, Func f) {
    arena_.execute([&] {
      tbb::parallel_for_each(std::forward<decltype(container)>(container), f);
    });
  }
  /**
   * @brief Invokes multiple functions in parallel.
   *
   * Runs all provided functions concurrently using `tbb::parallel_invoke`.
   *
   * @tparam Funcs Variadic list of callable types.
   * @param funcs Functions to invoke in parallel.
   */
  template <typename... Funcs>
  void parallel_invoke(Funcs&&... funcs) {
    arena_.execute(
        [&] { tbb::parallel_invoke(std::forward<Funcs>(funcs)...); });
  }

 private:
  /// Internal task arena for managing parallel execution.
  tbb::task_arena arena_;
};

}  // namespace core::math::diffeq::parallel