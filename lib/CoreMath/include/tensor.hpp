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

/**
 * @defgroup Tensor Math Tensor
 * @brief N-dimensional tensor implementation with common mathematical
 * operations
 * @{
 */

namespace core::math::tensor {

/**
 * @brief Compute the total size of a tensor given its shape.
 * @tparam Rank The rank (number of dimensions) of the tensor.
 * @param shape The shape of the tensor.
 * @return The total number of elements in the tensor.
 */
template <size_t Rank>
size_t compute_total_size(const std::array<size_t, Rank>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

/**
 * @brief Template class representing an N-dimensional mathematical tensor.
 * @tparam T The arithmetic type of tensor elements.
 * @tparam Rank The rank (number of dimensions) of the tensor.
 *
 * This class provides a generic implementation of mathematical tensors with
 * common operations like addition, reshaping, slicing, transposing, and more.
 */
template <typename T, size_t Rank>
  requires std::is_arithmetic_v<T>
class Tensor {
 private:
  std::vector<T> data_;             ///< Internal storage for tensor elements.
  std::array<size_t, Rank> shape_;  ///< Shape of the tensor.

  /**
   * @brief Compute the flattened index for the given multi-dimensional indices.
   * @param indices The multi-dimensional indices.
   * @return The flattened index.
   */
  size_t compute_flattened_index(
      const std::array<size_t, Rank>& indices) const {
    size_t idx = 0, stride = 1;
    for (int i = Rank - 1; i >= 0; --i) {
      idx += indices[i] * stride;
      stride *= shape_[i];
    }
    return idx;
  }

  /**
   * @brief Apply any pending lazy operations.
   */
  void apply_lazy() {
    if (lazy_op_) {
      data_ = lazy_op_.value()();
      lazy_op_.reset();
    }
  }

  std::optional<std::move_only_function<std::vector<T>()>>
      lazy_op_;  ///< Optional lazy operation.

 public:
  /**
   * @brief Construct a tensor with the given shape and initialize all elements
   * to a specified value.
   * @param shape The shape of the tensor.
   * @param init_value The value to initialize all elements with.
   */
  Tensor(const std::array<size_t, Rank>& shape, T init_value = T{})
      : shape_(shape), data_(compute_total_size(shape), init_value) {}

  /**
   * @brief Access elements of the tensor (mutable).
   * @param indices The multi-dimensional indices.
   * @return Reference to the requested element.
   */
  T& operator()(const std::array<size_t, Rank>& indices) {
    apply_lazy();
    return data_[compute_flattened_index(indices)];
  }

  /**
   * @brief Access elements of the tensor (const).
   * @param indices The multi-dimensional indices.
   * @return Const reference to the requested element.
   */
  const T& operator()(const std::array<size_t, Rank>& indices) const {
    return data_[compute_flattened_index(indices)];
  }

  /**
   * @brief Get the shape of the tensor.
   * @return The shape of the tensor.
   */
  const std::array<size_t, Rank>& shape() const { return shape_; }

  /**
   * @brief Add two tensors element-wise.
   * @param other The tensor to add.
   * @return A new tensor containing the result of the addition.
   * @throws std::invalid_argument if the shapes of the tensors do not match.
   */
  Tensor operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
      throw std::invalid_argument("Shapes do not match for addition.");
    }
    Tensor result(shape_);
    result.lazy_op_ = [this, &other]() {
      std::vector<T> res(data_.size());
      for (size_t i = 0; i < data_.size(); ++i) {
        res[i] = data_[i] + other.data_[i];
      }
      return res;
    };
    return result;
  }

  /**
   * @brief Reshape the tensor to a new shape.
   * @param new_shape The new shape of the tensor.
   * @return A new tensor with the specified shape.
   * @throws std::invalid_argument if the new shape does not match the total
   * size of the data.
   */
  Tensor reshape(const std::array<size_t, Rank>& new_shape) const {
    if (compute_total_size(shape_) != compute_total_size(new_shape)) {
      throw std::invalid_argument("New shape must match total size of data.");
    }
    Tensor result(new_shape);
    result.data_ = data_;
    return result;
  }

  /**
   * @brief Slice the tensor to create a sub-tensor.
   * @param start The starting indices of the slice.
   * @param end The ending indices of the slice (exclusive).
   * @return A new tensor containing the sliced data.
   * @throws std::invalid_argument if the slice range is invalid.
   */
  Tensor slice(const std::array<size_t, Rank>& start,
               const std::array<size_t, Rank>& end) const {
    std::array<size_t, Rank> new_shape;
    for (size_t i = 0; i < Rank; ++i) {
      if (end[i] <= start[i] || end[i] > shape_[i]) {
        throw std::invalid_argument("Invalid slice range.");
      }
      new_shape[i] = end[i] - start[i];
    }

    Tensor result(new_shape);
    auto total_size = compute_total_size(new_shape);
    for (size_t i = 0; i < total_size; ++i) {
      std::array<size_t, Rank> indices;
      size_t temp = i;
      for (int dim = Rank - 1; dim >= 0; --dim) {
        indices[dim] = temp % new_shape[dim] + start[dim];
        temp /= new_shape[dim];
      }
      result.data_[i] = (*this)(indices);
    }
    return result;
  }

  /**
   * @brief Transpose the tensor by swapping two axes.
   * @param axis1 The first axis to swap.
   * @param axis2 The second axis to swap.
   * @return A new tensor with the axes swapped.
   * @throws std::invalid_argument if the axes are invalid.
   */
  Tensor transpose(size_t axis1, size_t axis2) const {
    if (axis1 >= Rank || axis2 >= Rank) {
      throw std::invalid_argument("Invalid axes for transpose.");
    }

    auto new_shape = shape_;
    std::swap(new_shape[axis1], new_shape[axis2]);

    Tensor result(new_shape);
    auto total_size = compute_total_size(shape_);
    for (size_t i = 0; i < total_size; ++i) {
      std::array<size_t, Rank> indices;
      size_t temp = i;
      for (int dim = Rank - 1; dim >= 0; --dim) {
        indices[dim] = temp % shape_[dim];
        temp /= shape_[dim];
      }
      std::swap(indices[axis1], indices[axis2]);
      result(indices) = (*this)(indices);
    }
    return result;
  }

  /**
   * @brief Perform Singular Value Decomposition (SVD) on a 2D tensor.
   * @return A tuple containing the U, S, and V matrices.
   * @throws std::invalid_argument if the tensor is not 2D.
   * @throws std::runtime_error if the SVD does not converge.
   */
  std::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2>> svd() const {
    if (Rank != 2) {
      throw std::invalid_argument("SVD is only defined for 2D tensors.");
    }

    size_t m = shape_[0], n = shape_[1];
    Tensor<T, 2> U({m, m}, 0);
    Tensor<T, 2> S({m, n}, 0);
    Tensor<T, 2> V({n, n}, 0);

    Tensor<T, 2> B = *this;
    for (size_t i = 0; i < m; ++i) {
      U({i, i}) = 1;
    }
    for (size_t i = 0; i < n; ++i) {
      V({i, i}) = 1;
    }

    constexpr T tolerance = 1e-8;
    bool converged = false;
    size_t max_iterations = 1000;
    size_t iterations = 0;

    while (!converged && iterations < max_iterations) {
      iterations++;

      Tensor<T, 2> Q({n, n}, 0);
      Tensor<T, 2> R({m, n}, 0);

      for (size_t i = 0; i < n; ++i) {
        Q({i, i}) = 1;
      }

      for (size_t k = 0; k < n - 1; ++k) {
        T alpha = 0;
        for (size_t i = k; i < m; ++i) {
          alpha += B({i, k}) * B({i, k});
        }
        alpha = std::sqrt(alpha);

        if (alpha < tolerance) continue;

        Tensor<T, 2> H({m, m}, 0);
        for (size_t i = 0; i < m; ++i) {
          H({i, i}) = 1;
        }

        for (size_t i = k + 1; i < m; ++i) {
          T c = B({k, k}) / alpha;
          T s = B({i, k}) / alpha;

          for (size_t j = k; j < n; ++j) {
            T temp = c * B({k, j}) + s * B({i, j});
            B({i, j}) = -s * B({k, j}) + c * B({i, j});
            B({k, j}) = temp;
          }
        }
      }

      converged = true;
      for (size_t i = 0; i < std::min(m, n) - 1; ++i) {
        if (std::abs(B({i + 1, i})) > tolerance) {
          converged = false;
          break;
        }
      }
    }

    if (!converged) {
      throw std::runtime_error("SVD did not converge.");
    }

    for (size_t i = 0; i < std::min(m, n); ++i) {
      S({i, i}) = std::abs(B({i, i}));
    }

    return {U, S, V};
  }

  /**
   * @brief Perform LU decomposition on a square 2D tensor.
   * @return A tuple containing the L, U, and P matrices.
   * @throws std::invalid_argument if the tensor is not a square 2D tensor.
   */
  std::tuple<Tensor<T, 2>, Tensor<T, 2>, Tensor<T, 2>> lu() const {
    if (Rank != 2 || shape_[0] != shape_[1]) {
      throw std::invalid_argument(
          "LU decomposition is only defined for square 2D tensors.");
    }

    size_t n = shape_[0];
    Tensor<T, 2> L({n, n}, 0);
    Tensor<T, 2> U(*this);
    Tensor<T, 2> P({n, n}, 0);

    for (size_t i = 0; i < n; ++i) {
      P({i, i}) = 1;
    }

    for (size_t i = 0; i < n; ++i) {
      size_t pivot = i;
      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(U({j, i})) > std::abs(U({pivot, i}))) {
          pivot = j;
        }
      }
      if (pivot != i) {
        for (size_t k = 0; k < n; ++k) {
          std::swap(U({i, k}), U({pivot, k}));
          std::swap(P({i, k}), P({pivot, k}));
        }
      }

      for (size_t j = i + 1; j < n; ++j) {
        L({j, i}) = U({j, i}) / U({i, i});
        for (size_t k = i; k < n; ++k) {
          U({j, k}) -= L({j, i}) * U({i, k});
        }
      }
      L({i, i}) = 1;
    }

    return {L, U, P};
  }

  /**
   * @brief Print the tensor.
   */
  void print() const {
    auto print_recursive = [&](const auto& self, const auto& indices,
                               size_t dim) -> void {
      if (dim == Rank) {
        std::print("{} ", (*this)(indices));
        return;
      }
      std::print("[ ");
      for (size_t i = 0; i < shape_[dim]; ++i) {
        auto next_indices = indices;
        next_indices[dim] = i;
        self(self, next_indices, dim + 1);
      }
      std::print("] ");
    };
    print_recursive(print_recursive, std::array<size_t, Rank>{}, 0);
    std::print("\n");
  }
};

}  // namespace core::math::tensor

/** @} */  // end of Tensor group
