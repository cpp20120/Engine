#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <numeric>
#include <optional>
#include <print>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../parallel/parallel_executor.hpp"
#include "../tensor.hpp"

namespace core::math::tensor::api {

/**
 * @class TensorAPI
 * @brief High-level API for tensor operations.
 */
class TensorAPI {
 public:
  // Tensor Creation

  /**
   * @brief Creates a tensor with the specified shape and initial value.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param shape Shape of the tensor.
   * @param init_value Initial value for tensor elements.
   * @return Tensor with the specified shape and initial value.
   */
  template <typename T, size_t Rank>
  static auto create(const std::array<size_t, Rank>& shape,
                     T init_value = T{}) {
    return core::math::tensor::Tensor<T, Rank>(shape, init_value);
  }

  // Tensor Operations

  /**
   * @brief Adds two tensors element-wise.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param a First tensor.
   * @param b Second tensor.
   * @return Result of the addition.
   * @throws std::invalid_argument if the shapes of the tensors do not match.
   */
  template <typename T, size_t Rank>
  static auto add(const core::math::tensor::Tensor<T, Rank>& a,
                  const core::math::tensor::Tensor<T, Rank>& b) {
    return a + b;
  }

  /**
   * @brief Reshapes the tensor to a new shape.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param tensor Tensor to reshape.
   * @param new_shape New shape of the tensor.
   * @return Tensor with the specified new shape.
   * @throws std::invalid_argument if the new shape does not match the total
   * size of the data.
   */
  template <typename T, size_t Rank>
  static auto reshape(const core::math::tensor::Tensor<T, Rank>& tensor,
                      const std::array<size_t, Rank>& new_shape) {
    return tensor.reshape(new_shape);
  }

  /**
   * @brief Slices the tensor to create a sub-tensor.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param tensor Tensor to slice.
   * @param start Starting indices of the slice.
   * @param end Ending indices of the slice (exclusive).
   * @return Tensor containing the sliced data.
   * @throws std::invalid_argument if the slice range is invalid.
   */
  template <typename T, size_t Rank>
  static auto slice(const core::math::tensor::Tensor<T, Rank>& tensor,
                    const std::array<size_t, Rank>& start,
                    const std::array<size_t, Rank>& end) {
    return tensor.slice(start, end);
  }

  /**
   * @brief Transposes the tensor by swapping two axes.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param tensor Tensor to transpose.
   * @param axis1 First axis to swap.
   * @param axis2 Second axis to swap.
   * @return Tensor with the axes swapped.
   * @throws std::invalid_argument if the axes are invalid.
   */
  template <typename T, size_t Rank>
  static auto transpose(const core::math::tensor::Tensor<T, Rank>& tensor,
                        size_t axis1, size_t axis2) {
    return tensor.transpose(axis1, axis2);
  }

  /**
   * @brief Performs Singular Value Decomposition (SVD) on a 2D tensor.
   * @tparam T Type of tensor elements.
   * @param tensor Tensor to decompose.
   * @return A tuple containing the U, S, and V matrices.
   * @throws std::invalid_argument if the tensor is not 2D.
   * @throws std::runtime_error if the SVD does not converge.
   */
  template <typename T>
  static auto svd(const core::math::tensor::Tensor<T, 2>& tensor) {
    return tensor.svd();
  }

  /**
   * @brief Performs LU decomposition on a square 2D tensor.
   * @tparam T Type of tensor elements.
   * @param tensor Tensor to decompose.
   * @return A tuple containing the L, U, and P matrices.
   * @throws std::invalid_argument if the tensor is not a square 2D tensor.
   */
  template <typename T>
  static auto lu(const core::math::tensor::Tensor<T, 2>& tensor) {
    return tensor.lu();
  }

  /**
   * @brief Prints the tensor.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param tensor Tensor to print.
   */
  template <typename T, size_t Rank>
  static void print(const core::math::tensor::Tensor<T, Rank>& tensor) {
    tensor.print();
  }

  // Parallel Tensor Operations

  /**
   * @brief Adds two tensors element-wise in parallel.
   * @tparam T Type of tensor elements.
   * @tparam Rank Rank of the tensor.
   * @param result Result tensor.
   * @param a First tensor.
   * @param b Second tensor.
   */
  template <typename T, size_t Rank>
  static void parallel_add(core::math::tensor::Tensor<T, Rank>& result,
                           const core::math::tensor::Tensor<T, Rank>& a,
                           const core::math::tensor::Tensor<T, Rank>& b) {
    core::math::tensor::parallel::parallel_tensor_add(result, a, b);
  }
};

}  // namespace core::math::tensor::api
