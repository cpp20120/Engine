#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_group.h>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace core::math::parallel {

/**
 * @brief Executes a function in parallel over a range of indices.
 *
 * This function uses Intel TBB's parallel_for to execute the given function
 * over a specified range of indices. It divides the range into smaller chunks
 * and processes each chunk in parallel.
 *
 * @tparam IndexType Type of the indices.
 * @tparam Function Type of the function to execute.
 * @param begin Start index of the range.
 * @param end End index of the range.
 * @param func Function to execute for each index.
 */
template <typename IndexType, typename Function>
void parallel_for(IndexType begin, IndexType end, Function&& func) {
  tbb::parallel_for(tbb::blocked_range<IndexType>(begin, end),
                    [&func](const tbb::blocked_range<IndexType>& range) {
                      for (IndexType i = range.begin(); i != range.end(); ++i) {
                        func(i);
                      }
                    });
}

/**
 * @brief Performs a parallel reduction over a range of indices.
 *
 * This function uses Intel TBB's parallel_reduce to perform a reduction
 * operation over a specified range of indices. It divides the range into
 * smaller chunks, processes each chunk in parallel, and combines the results
 * using the reduction function.
 *
 * @tparam IndexType Type of the indices.
 * @tparam ValueType Type of the values.
 * @tparam Function Type of the function to apply.
 * @tparam Reduction Type of the reduction function.
 * @param begin Start index of the range.
 * @param end End index of the range.
 * @param func Function to apply for each index.
 * @param reduce Reduction function.
 * @param identity Identity value for the reduction.
 * @return Result of the reduction.
 */
template <typename IndexType, typename ValueType, typename Function,
          typename Reduction>
ValueType parallel_reduce(IndexType begin, IndexType end, Function&& func,
                          Reduction&& reduce, ValueType identity) {
  return tbb::parallel_reduce(
      tbb::blocked_range<IndexType>(begin, end), identity,
      [&func](const tbb::blocked_range<IndexType>& range, ValueType init) {
        for (IndexType i = range.begin(); i != range.end(); ++i) {
          init = reduce(init, func(i));
        }
        return init;
      },
      reduce);
}

/**
 * @brief Executes multiple functions in parallel.
 *
 * This function uses Intel TBB's task_group to execute multiple functions in
 * parallel. Each function is run as a separate task within the task group.
 *
 * @tparam Function Type of the functions to execute.
 * @param functions Vector of functions to execute.
 */
template <typename Function>
void parallel_invoke(std::vector<Function>&& functions) {
  tbb::task_group group;
  for (auto& func : functions) {
    group.run(func);
  }
  group.wait();
}

}  // namespace core::math::parallel
