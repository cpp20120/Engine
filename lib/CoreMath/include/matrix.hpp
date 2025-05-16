#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_group.h>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "./aallocator.hpp"
#include "./vector.hpp"

/**
 * @namespace core::math::parallel
 * @brief Namespace for parallel matrix operations.
 */
namespace core::math::parallel {

/**
 * @brief Executes a function in parallel over a range of indices.
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

/**
 * @defgroup core_math_matrix Core Math Matrix
 * @brief Core Matrix library implementation
 * @{
 */

namespace core::math::matrix {

/**
 * @brief Concept for arithmetic types supporting basic operations
 *
 * Requires types to support:
 * - Standard arithmetic operations (+, -, *, /)
 * - Unary negation
 * - std::is_arithmetic_v trait
 *
 * @tparam T Type to check
 */
template <typename T>
concept ArithmeticValue = requires(T a, T b) {
  requires std::is_arithmetic_v<T>;
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
  { a / b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
};

/**
 * @brief Matrix class for linear algebra operations.
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
class Matrix {
 private:
  // Use std::array for small matrices, std::vector with aligned allocator for
  // large ones
  using StorageType = std::conditional_t<
      (Rows * Cols <= 16), std::array<T, Rows * Cols>,
      std::vector<T, core::math::alloc::AlignedAllocator<T>>>;
  StorageType data_;

 public:
  /**
   * @brief Default constructor initializes matrix to zeros
   */
  constexpr Matrix() {
    if constexpr (Rows * Cols <= 16) {
      data_.fill(T{});
    } else {
      data_.resize(Rows * Cols, T{});
    }
  }

  /**
   * @brief Construct from initializer list.
   * @param init Initializer list of matrix elements.
   * @throws std::invalid_argument If initializer list size does not match
   * matrix dimensions.
   */
  constexpr Matrix(std::initializer_list<T> init) {
    if (init.size() != Rows * Cols) {
      throw std::invalid_argument(
          "Initializer list size does not match matrix dimensions.");
    }
    if constexpr (Rows * Cols <= 16) {
      std::copy(init.begin(), init.end(), data_.begin());
    } else {
      data_.assign(init.begin(), init.end());
    }
  }

  /**
   * @brief Move constructor
   */
  Matrix(Matrix&& other) noexcept = default;

  /**
   * @brief Copy constructor
   */
  Matrix(const Matrix& other) = default;

  /**
   * @brief Access matrix element (mutable)
   */
  constexpr T& operator()(size_t i, size_t j) { return data_[i * Cols + j]; }

  /**
   * @brief Access matrix element (const)
   */
  constexpr const T& operator()(size_t i, size_t j) const {
    return data_[i * Cols + j];
  }

  /**
   * @brief Get pointer to underlying data
   */
  constexpr T* data() noexcept {
    if constexpr (Rows * Cols <= 16) {
      return data_.data();
    } else {
      return data_.data();
    }
  }

  /**
   * @brief Get const pointer to underlying data
   */
  constexpr const T* data() const noexcept {
    if constexpr (Rows * Cols <= 16) {
      return data_.data();
    } else {
      return data_.data();
    }
  }

  /**
   * @brief Get matrix dimensions
   */
  constexpr std::pair<size_t, size_t> size() const noexcept {
    return {Rows, Cols};
  }

  /**
   * @brief Check if matrix is square
   */
  constexpr bool is_square() const noexcept { return Rows == Cols; }

  /**
   * @brief Print matrix to standard output
   */
  constexpr void print() const {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        std::print("{} ", (*this)(i, j));
      }
      std::print("\n");
    }
  }

  /**
   * @brief Create identity matrix
   */
  static constexpr Matrix identity() {
    static_assert(Rows == Cols, "Identity matrix must be square.");
    Matrix result;
    for (size_t i = 0; i < Rows; ++i) {
      result(i, i) = T(1);
    }
    return result;
  }

  /**
   * @brief Set all elements to zero
   */
  constexpr void zero() {
    if constexpr (Rows * Cols <= 16) {
      data_.fill(T{});
    } else {
      std::fill(data_.begin(), data_.end(), T{});
    }
  }

  // Iterators
  constexpr auto begin() noexcept { return data_.begin(); }
  constexpr auto end() noexcept { return data_.end(); }
  constexpr auto begin() const noexcept { return data_.begin(); }
  constexpr auto end() const noexcept { return data_.end(); }
  constexpr auto cbegin() const noexcept { return data_.cbegin(); }
  constexpr auto cend() const noexcept { return data_.cend(); }

  /**
   * @brief Check if a matrix is orthogonal.
   * @tparam T Arithmetic type of matrix elements.
   * @tparam N Number of rows and columns in the matrix.
   * @param m Matrix to check.
   * @return True if the matrix is orthogonal, false otherwise.
   */
  template <ArithmeticValue T, size_t N>
  bool is_orthogonal(const Matrix<T, N, N>& m) {
    auto mt = transpose(m);
    auto identity = Matrix<T, N, N>::identity();
    return multiply(m, mt) == identity;
  }

  /**
   * @brief Create a compile-time submatrix slice
   * @tparam StartRow Start row index.
   * @tparam EndRow End row index.
   * @tparam StartCol Start column index.
   * @tparam EndCol End column index.
   * @return Submatrix slice.
   */
  template <size_t StartRow, size_t EndRow, size_t StartCol, size_t EndCol>
  Matrix<T, EndRow - StartRow, EndCol - StartCol> slice() const {
    static_assert(StartRow < EndRow && EndRow <= Rows);
    static_assert(StartCol < EndCol && EndCol <= Cols);
    Matrix<T, EndRow - StartRow, EndCol - StartCol> result;
    for (size_t i = StartRow; i < EndRow; ++i) {
      for (size_t j = StartCol; j < EndCol; ++j) {
        result(i - StartRow, j - StartCol) = (*this)(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Create a dynamic submatrix slice
   * @param start_row Start row index.
   * @param end_row End row index.
   * @param start_col Start column index.
   * @param end_col End column index.
   * @return Submatrix slice.
   * @throws std::invalid_argument If slice dimensions are invalid.
   */
  Matrix<T, Rows, Cols> slice(size_t start_row, size_t end_row,
                              size_t start_col, size_t end_col) const {
    if (start_row >= end_row || end_row > Rows || start_col >= end_col ||
        end_col > Cols) {
      throw std::invalid_argument("Invalid slice dimensions.");
    }
    Matrix<T, Rows, Cols> result;
    result.zero();
    for (size_t i = start_row; i < end_row; ++i) {
      for (size_t j = start_col; j < end_col; ++j) {
        result(i - start_row, j - start_col) = (*this)(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Move assignment operator
   */
  Matrix& operator=(Matrix&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Matrix& operator=(const Matrix& other) = default;
};

/**
 * @brief Matrix View for non-owning submatrices
 * @tparam T Arithmetic type of matrix elements.
 */
template <ArithmeticValue T>
class MatrixView {
 private:
  T* data_;
  size_t rows_, cols_, stride_;

 public:
  /**
   * @brief Constructor for MatrixView.
   * @param data Pointer to the data.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param stride Stride of the matrix.
   */
  MatrixView(T* data, size_t rows, size_t cols, size_t stride)
      : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

  /**
   * @brief Access matrix element (mutable)
   * @param i Row index.
   * @param j Column index.
   * @return Reference to the matrix element.
   */
  T& operator()(size_t i, size_t j) { return data_[i * stride_ + j]; }

  /**
   * @brief Access matrix element (const)
   * @param i Row index.
   * @param j Column index.
   * @return Const reference to the matrix element.
   */
  const T& operator()(size_t i, size_t j) const {
    return data_[i * stride_ + j];
  }

  /**
   * @brief Get matrix dimensions
   * @return Pair of rows and columns.
   */
  std::pair<size_t, size_t> size() const { return {rows_, cols_}; }
};

// Expression templates
template <typename Lhs, typename Rhs>
struct MatrixAddExpr {
  const Lhs& lhs;
  const Rhs& rhs;
  MatrixAddExpr(const Lhs& l, const Rhs& r) : lhs(l), rhs(r) {}
  auto operator()(size_t i, size_t j) const { return lhs(i, j) + rhs(i, j); }
  std::pair<size_t, size_t> size() const { return lhs.size(); }
};

template <typename Lhs, typename Rhs>
struct MatrixMultiplyExpr {
  const Lhs& lhs;
  const Rhs& rhs;
  MatrixMultiplyExpr(const Lhs& l, const Rhs& r) : lhs(l), rhs(r) {}
  auto operator()(size_t i, size_t j) const {
    auto sum = {};
    for (size_t k = 0; k < lhs.size().second; ++k) {
      sum += lhs(i, k) * rhs(k, j);
    }
    return sum;
  }
  std::pair<size_t, size_t> size() const {
    return {lhs.size().first, rhs.size().second};
  }
};

// Evaluate expression to Matrix
template <ArithmeticValue T, size_t Rows, size_t Cols, typename Expr>
Matrix<T, Rows, Cols> evaluate(const Expr& expr) {
  Matrix<T, Rows, Cols> result;
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      result(i, j) = expr(i, j);
    }
  }
  return result;
}

/**
 * @brief Matrix addition
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param a First matrix.
 * @param b Second matrix.
 * @return Result of the addition.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> add(const Matrix<T, Rows, Cols>& a,
                                    const Matrix<T, Rows, Cols>& b) {
  if constexpr (Rows * Cols <= 4) {  // Unrolled for small matrices
    Matrix<T, Rows, Cols> result;
    result(0, 0) = a(0, 0) + b(0, 0);
    result(0, 1) = a(0, 1) + b(0, 1);
    result(1, 0) = a(1, 0) + b(1, 0);
    result(1, 1) = a(1, 1) + b(1, 1);
    return result;
  } else {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = a(i, j) + b(i, j);
      }
    }
    return result;
  }
}

/**
 * @brief Matrix subtraction
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param a First matrix.
 * @param b Second matrix.
 * @return Result of the subtraction.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> subtract(const Matrix<T, Rows, Cols>& a,
                                         const Matrix<T, Rows, Cols>& b) {
  Matrix<T, Rows, Cols> result;
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      result(i, j) = a(i, j) - b(i, j);
    }
  }
  return result;
}

/**
 * @brief Scalar multiplication
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to multiply.
 * @param scalar Scalar value.
 * @return Result of the multiplication.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> scalar_multiply(
    const Matrix<T, Rows, Cols>& matrix, T scalar) {
  Matrix<T, Rows, Cols> result;
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      result(i, j) = matrix(i, j) * scalar;
    }
  }
  return result;
}

/**
 * @brief Matrix transpose
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to transpose.
 * @return Transposed matrix.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Cols, Rows> transpose(const Matrix<T, Rows, Cols>& matrix) {
  Matrix<T, Cols, Rows> result;
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      result(j, i) = matrix(i, j);
    }
  }
  return result;
}

/**
 * @brief Matrix determinant
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the determinant of.
 * @return Determinant of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr T determinant(const Matrix<T, N, N>& matrix) {
  if constexpr (N == 1) {
    return matrix(0, 0);
  } else if constexpr (N == 2) {
    return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
  } else if constexpr (N == 3) {
    return matrix(0, 0) * matrix(1, 1) * matrix(2, 2) +
           matrix(0, 1) * matrix(1, 2) * matrix(2, 0) +
           matrix(0, 2) * matrix(1, 0) * matrix(2, 1) -
           matrix(0, 2) * matrix(1, 1) * matrix(2, 0) -
           matrix(0, 0) * matrix(1, 2) * matrix(2, 1) -
           matrix(0, 1) * matrix(1, 0) * matrix(2, 2);
  } else {
    return determinant_via_lu(matrix);
  }
}

/**
 * @brief Determinant via LU decomposition
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the determinant of.
 * @return Determinant of the matrix.
 */
template <typename T, size_t N>
T determinant_via_lu(const Matrix<T, N, N>& matrix) {
  auto [L, U, P] = lu_decomposition(matrix);
  T det = T(1);
  for (size_t i = 0; i < N; ++i) {
    if (std::abs(U(i, i)) < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error("Matrix is singular or nearly singular.");
    }
    det *= U(i, i);
  }
  size_t num_swaps = 0;
  for (size_t i = 0; i < N; ++i) {
    if (P(i, i) != T(1)) {
      num_swaps++;
    }
  }
  if (num_swaps % 2 != 0) {
    det = -det;
  }
  return det;
}

/**
 * @brief Block matrix multiplication
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the result matrix.
 * @tparam Inner Number of columns in the first matrix and rows in the second
 * matrix.
 * @tparam Cols Number of columns in the result matrix.
 * @param result Result matrix.
 * @param a First matrix.
 * @param b Second matrix.
 * @param block_size Size of the blocks.
 */
template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
void block_multiply(Matrix<T, Rows, Cols>& result,
                    const Matrix<T, Rows, Inner>& a,
                    const Matrix<T, Inner, Cols>& b, size_t block_size = 64) {
  result.zero();
  for (size_t i = 0; i < Rows; i += block_size) {
    for (size_t j = 0; j < Cols; j += block_size) {
      for (size_t k = 0; k < Inner; k += block_size) {
        for (size_t ii = i; ii < std::min(i + block_size, Rows); ++ii) {
          for (size_t jj = j; jj < std::min(j + block_size, Cols); ++jj) {
            T sum = result(ii, jj);
            for (size_t kk = k; kk < std::min(k + block_size, Inner); ++kk) {
              sum += a(ii, kk) * b(kk, jj);
            }
            result(ii, jj) = sum;
          }
        }
      }
    }
  }
}

/**
 * @brief Matrix multiplication
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the result matrix.
 * @tparam Inner Number of columns in the first matrix and rows in the second
 * matrix.
 * @tparam Cols Number of columns in the result matrix.
 * @param a First matrix.
 * @param b Second matrix.
 * @return Result of the multiplication.
 */
template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
constexpr Matrix<T, Rows, Cols> multiply(const Matrix<T, Rows, Inner>& a,
                                         const Matrix<T, Inner, Cols>& b) {
  Matrix<T, Rows, Cols> result;
  if constexpr (Rows * Cols * Inner <= 64) {  // Unrolled for small matrices
    block_multiply(result, a, b, 16);
  } else {
    block_multiply(result, a, b);
  }
  return result;
}

/**
 * @brief Matrix inverse using LU decomposition
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to invert.
 * @return Inverse of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, N> inverse(const Matrix<T, N, N>& matrix) {
  auto [L, U, P] = lu_decomposition(matrix);
  Matrix<T, N, N> inv = Matrix<T, N, N>::identity();
  for (size_t col = 0; col < N; ++col) {
    std::array<T, N> y{};
    for (size_t i = 0; i < N; ++i) {
      T sum = P(col, i);
      for (size_t j = 0; j < i; ++j) {
        sum -= L(i, j) * y[j];
      }
      if (std::abs(L(i, i)) < std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("Matrix is singular or nearly singular.");
      }
      y[i] = sum / L(i, i);
    }
    for (size_t i = N; i-- > 0;) {
      T sum = y[i];
      for (size_t j = i + 1; j < N; ++j) {
        sum -= U(i, j) * inv(j, col);
      }
      if (std::abs(U(i, i)) < std::numeric_limits<T>::epsilon()) {
        throw std::runtime_error("Matrix is singular or nearly singular.");
      }
      inv(i, col) = sum / U(i, i);
    }
  }
  return inv;
}

/**
 * @brief Operator overloads
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& a,
                                          const Matrix<T, Rows, Cols>& b) {
  return evaluate<T, Rows, Cols>(MatrixAddExpr(a, b));
}

template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& a,
                                          const Matrix<T, Rows, Cols>& b) {
  return subtract(a, b);
}

template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, Cols>& matrix,
                                          T scalar) {
  return scalar_multiply(matrix, scalar);
}

template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(T scalar,
                                          const Matrix<T, Rows, Cols>& matrix) {
  return matrix * scalar;
}

template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, Inner>& a,
                                          const Matrix<T, Inner, Cols>& b) {
  return multiply(a, b);
}

template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr bool operator==(const Matrix<T, Rows, Cols>& a,
                          const Matrix<T, Rows, Cols>& b) {
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      if (std::abs(a(i, j) - b(i, j)) > std::numeric_limits<T>::epsilon()) {
        return false;
      }
    }
  }
  return true;
}

template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr bool operator!=(const Matrix<T, Rows, Cols>& a,
                          const Matrix<T, Rows, Cols>& b) {
  return !(a == b);
}

/**
 * @brief Matrix trace
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the trace of.
 * @return Trace of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr T trace(const Matrix<T, N, N>& matrix) {
  T sum = T(0);
  for (size_t i = 0; i < N; ++i) {
    sum += matrix(i, i);
  }
  return sum;
}

/**
 * @brief Matrix minor
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the minor of.
 * @param row Row index.
 * @param col Column index.
 * @return Minor of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N - 1, N - 1> minor(const Matrix<T, N, N>& matrix,
                                        size_t row, size_t col) {
  Matrix<T, N - 1, N - 1> result;
  size_t r = 0, c = 0;
  for (size_t i = 0; i < N; ++i) {
    if (i == row) continue;
    c = 0;
    for (size_t j = 0; j < N; ++j) {
      if (j == col) continue;
      result(r, c) = matrix(i, j);
      ++c;
    }
    ++r;
  }
  return result;
}

/**
 * @brief Matrix cofactor
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the cofactor of.
 * @param row Row index.
 * @param col Column index.
 * @return Cofactor of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr T cofactor(const Matrix<T, N, N>& matrix, size_t row, size_t col) {
  return ((row + col) % 2 == 0 ? T(1) : T(-1)) *
         determinant(minor(matrix, row, col));
}

/**
 * @brief Cofactor matrix
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the cofactor matrix of.
 * @return Cofactor matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, N> cofactor_matrix(const Matrix<T, N, N>& matrix) {
  Matrix<T, N, N> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result(i, j) = cofactor(matrix, i, j);
    }
  }
  return result;
}

/**
 * @brief Matrix adjugate
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param matrix Matrix to compute the adjugate of.
 * @return Adjugate of the matrix.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, N> adjugate(const Matrix<T, N, N>& matrix) {
  return transpose(cofactor_matrix(matrix));
}

/**
 * @brief Matrix rank
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to compute the rank of.
 * @return Rank of the matrix.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr size_t rank(Matrix<T, Rows, Cols> matrix) {
  size_t rank = 0;
  for (size_t row = 0; row < Rows; ++row) {
    size_t leading_col = row;
    while (leading_col < Cols && std::abs(matrix(row, leading_col)) <
                                     std::numeric_limits<T>::epsilon()) {
      ++leading_col;
    }
    if (leading_col == Cols) continue;
    ++rank;
    for (size_t i = row + 1; i < Rows; ++i) {
      T factor = matrix(i, leading_col) / matrix(row, leading_col);
      for (size_t j = leading_col; j < Cols; ++j) {
        matrix(i, j) -= factor * matrix(row, j);
      }
    }
  }
  return rank;
}

/**
 * @brief Get matrix row as array
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to get the row from.
 * @param r Row index.
 * @return Array representing the row.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr std::array<T, Cols> get_row(const Matrix<T, Rows, Cols>& matrix,
                                      size_t r) {
  std::array<T, Cols> result{};
  for (size_t j = 0; j < Cols; ++j) {
    result[j] = matrix(r, j);
  }
  return result;
}

/**
 * @brief Get matrix column as array
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to get the column from.
 * @param c Column index.
 * @return Array representing the column.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr std::array<T, Rows> get_column(const Matrix<T, Rows, Cols>& matrix,
                                         size_t c) {
  std::array<T, Rows> result{};
  for (size_t i = 0; i < Rows; ++i) {
    result[i] = matrix(i, c);
  }
  return result;
}

/**
 * @brief Fill matrix with random values
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to fill with random values.
 * @param min Minimum value.
 * @param max Maximum value.
 */
template <typename T, size_t Rows, size_t Cols>
void randomize(Matrix<T, Rows, Cols>& matrix, T min, T max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(min, max);
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        matrix(i, j) = dist(gen);
      }
    }
  } else {
    std::uniform_real_distribution<T> dist(min, max);
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        matrix(i, j) = dist(gen);
      }
    }
  }
}

/**
 * @brief Fill matrix with random values
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to fill with random values.
 */
template <typename T, size_t Rows, size_t Cols>
void randomize(Matrix<T, Rows, Cols>& matrix) {
  randomize(matrix, T(0), T(1));
}

/**
 * @brief Fill matrix with random values
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param matrix Matrix to fill with random values.
 * @param range Range of values.
 */
template <typename T, size_t Rows, size_t Cols>
void randomize(Matrix<T, Rows, Cols>& matrix, T range) {
  randomize(matrix, -range, range);
}

/**
 * @brief Fill matrix with random values using a custom generator
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @tparam Generator Type of the random generator.
 * @param matrix Matrix to fill with random values.
 * @param min Minimum value.
 * @param max Maximum value.
 * @param gen Random generator.
 */
template <typename T, size_t Rows, size_t Cols, typename Generator>
void randomize(Matrix<T, Rows, Cols>& matrix, T min, T max, Generator& gen) {
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(min, max);
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        matrix(i, j) = dist(gen);
      }
    }
  } else {
    std::uniform_real_distribution<T> dist(min, max);
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        matrix(i, j) = dist(gen);
      }
    }
  }
}

/**
 * @brief QR decomposition using Householder method
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param A Matrix to decompose.
 * @return Pair of Q and R matrices.
 */
template <ArithmeticValue T, size_t N>
constexpr std::pair<Matrix<T, N, N>, Matrix<T, N, N>> qr_decomposition(
    const Matrix<T, N, N>& A) {
  Matrix<T, N, N> Q = Matrix<T, N, N>::identity();
  Matrix<T, N, N> R = A;
  for (size_t k = 0; k < N - 1; ++k) {
    T norm = T(0);
    for (size_t i = k; i < N; ++i) {
      norm += R(i, k) * R(i, k);
    }
    norm = std::sqrt(norm);
    T alpha = -std::copysign(norm, R(k, k));
    T r = std::sqrt(T(0.5) * (alpha * alpha - R(k, k) * alpha));
    std::array<T, N> v{};
    v[k] = (R(k, k) - alpha) / (T(2) * r);
    for (size_t i = k + 1; i < N; ++i) {
      v[i] = R(i, k) / (T(2) * r);
    }
    for (size_t j = k; j < N; ++j) {
      T dot = T(0);
      for (size_t i = k; i < N; ++i) {
        dot += v[i] * R(i, j);
      }
      for (size_t i = k; i < N; ++i) {
        R(i, j) -= T(2) * v[i] * dot;
      }
    }
    for (size_t j = 0; j < N; ++j) {
      T dot = T(0);
      for (size_t i = k; i < N; ++i) {
        dot += v[i] * Q(j, i);
      }
      for (size_t i = k; i < N; ++i) {
        Q(j, i) -= T(2) * v[i] * dot;
      }
    }
  }
  return std::make_pair(transpose(Q), R);
}

/**
 * @brief LU decomposition using Doolittle's method
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows and columns in the matrix.
 * @param A Matrix to decompose.
 * @return Tuple of L, U, and P matrices.
 */
template <ArithmeticValue T, size_t N>
constexpr std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>>
lu_decomposition(const Matrix<T, N, N>& A) {
  Matrix<T, N, N> L = Matrix<T, N, N>::identity();
  Matrix<T, N, N> U;
  Matrix<T, N, N> P = Matrix<T, N, N>::identity();
  Matrix<T, N, N> A_copy = A;

  for (size_t k = 0; k < N; ++k) {
    size_t max_row = k;
    T max_val = std::abs(A_copy(k, k));
    for (size_t i = k + 1; i < N; ++i) {
      if (std::abs(A_copy(i, k)) > max_val) {
        max_val = std::abs(A_copy(i, k));
        max_row = i;
      }
    }
    if (std::abs(max_val) < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error("Matrix is singular or nearly singular.");
    }
    if (max_row != k) {
      for (size_t j = 0; j < N; ++j) {
        std::swap(P(k, j), P(max_row, j));
        std::swap(A_copy(k, j), A_copy(max_row, j));
      }
    }
    U(k, k) = A_copy(k, k);
    for (size_t i = k + 1; i < N; ++i) {
      L(i, k) = A_copy(i, k) / U(k, k);
      A_copy(i, k) = T(0);
    }
    for (size_t j = k + 1; j < N; ++j) {
      U(k, j) = A_copy(k, j);
      for (size_t i = k + 1; i < N; ++i) {
        A_copy(i, j) -= L(i, k) * U(k, j);
      }
    }
  }
  return std::make_tuple(L, U, P);
}

/**
 * @brief Dot product for vectors (1-column matrices)
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows in the matrix.
 * @param a First vector.
 * @param b Second vector.
 * @return Dot product of the vectors.
 */
template <ArithmeticValue T, size_t N>
constexpr T dot(const Matrix<T, N, 1>& a, const Matrix<T, N, 1>& b) {
  T sum = T(0);
  for (size_t i = 0; i < N; ++i) {
    sum += a(i, 0) * b(i, 0);
  }
  return sum;
}

/**
 * @brief Norm for vectors (1-column matrices)
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows in the matrix.
 * @param v Vector to compute the norm of.
 * @return Norm of the vector.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, 1> normalize(const Matrix<T, N, 1>& v) {
  T length = norm(v);
  if (std::abs(length) < std::numeric_limits<T>::epsilon()) {
    return v;
  }
  Matrix<T, N, 1> result;
  for (size_t i = 0; i < N; ++i) {
    result(i, 0) = v(i, 0) / length;
  }
  return result;
}

/**
 * @namespace transform
 * @brief Geometric transformation matrices
 */
namespace transform {
/**
 * @brief Create a 2D translation matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param x Translation in x direction.
 * @param y Translation in y direction.
 * @return Translation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> translate2d(T x, T y) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  m(0, 2) = x;
  m(1, 2) = y;
  return m;
}

/**
 * @brief Create a 2D rotation matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param angle_rad Rotation angle in radians.
 * @return Rotation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> rotate2d(T angle_rad) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  T c = std::cos(angle_rad);
  T s = std::sin(angle_rad);
  m(0, 0) = c;
  m(0, 1) = -s;
  m(1, 0) = s;
  m(1, 1) = c;
  return m;
}

/**
 * @brief Create a 2D scaling matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param sx Scaling factor in x direction.
 * @param sy Scaling factor in y direction.
 * @return Scaling matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> scale2d(T sx, T sy) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  m(0, 0) = sx;
  m(1, 1) = sy;
  return m;
}

/**
 * @brief Create a 2D reflection matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param x_axis Whether to reflect over the x-axis.
 * @param y_axis Whether to reflect over the y-axis.
 * @return Reflection matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> reflect2d(bool x_axis, bool y_axis) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  if (x_axis) m(1, 1) = -1;
  if (y_axis) m(0, 0) = -1;
  return m;
}

/**
 * @brief Create a 3D translation matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param x Translation in x direction.
 * @param y Translation in y direction.
 * @param z Translation in z direction.
 * @return Translation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> translate3d(T x, T y, T z) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  m(0, 3) = x;
  m(1, 3) = y;
  m(2, 3) = z;
  return m;
}

/**
 * @brief Create a 3D rotation matrix around the x-axis
 * @tparam T Arithmetic type of matrix elements.
 * @param angle_rad Rotation angle in radians.
 * @return Rotation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> rotate3d_x(T angle_rad) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  T c = std::cos(angle_rad);
  T s = std::sin(angle_rad);
  m(1, 1) = c;
  m(1, 2) = -s;
  m(2, 1) = s;
  m(2, 2) = c;
  return m;
}

/**
 * @brief Create a 3D rotation matrix around the y-axis
 * @tparam T Arithmetic type of matrix elements.
 * @param angle_rad Rotation angle in radians.
 * @return Rotation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> rotate3d_y(T angle_rad) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  T c = std::cos(angle_rad);
  T s = std::sin(angle_rad);
  m(0, 0) = c;
  m(0, 2) = s;
  m(2, 0) = -s;
  m(2, 2) = c;
  return m;
}

/**
 * @brief Create a 3D rotation matrix around the z-axis
 * @tparam T Arithmetic type of matrix elements.
 * @param angle_rad Rotation angle in radians.
 * @return Rotation matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> rotate3d_z(T angle_rad) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  T c = std::cos(angle_rad);
  T s = std::sin(angle_rad);
  m(0, 0) = c;
  m(0, 1) = -s;
  m(1, 0) = s;
  m(1, 1) = c;
  return m;
}

/**
 * @brief Create a 3D scaling matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param sx Scaling factor in x direction.
 * @param sy Scaling factor in y direction.
 * @param sz Scaling factor in z direction.
 * @return Scaling matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> scale3d(T sx, T sy, T sz) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  m(0, 0) = sx;
  m(1, 1) = sy;
  m(2, 2) = sz;
  return m;
}

/**
 * @brief Create a 3D reflection matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param x_axis Whether to reflect over the x-axis.
 * @param y_axis Whether to reflect over the y-axis.
 * @param z_axis Whether to reflect over the z-axis.
 * @return Reflection matrix.
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> reflect3d(bool x_axis, bool y_axis, bool z_axis) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  if (x_axis) m(0, 0) = -1;
  if (y_axis) m(1, 1) = -1;
  if (z_axis) m(2, 2) = -1;
  return m;
}

/**
 * @brief Create a perspective projection matrix
 * @tparam T Arithmetic type of matrix elements.
 * @param fov Field of view in radians.
 * @param aspect Aspect ratio.
 * @param near Near clipping plane.
 * @param far Far clipping plane.
 * @return Perspective projection matrix.
 */
template <ArithmeticValue T>
Matrix<T, 4, 4> perspective(T fov, T aspect, T near, T far) {
  Matrix<T, 4, 4> m;
  T tan_half_fov = std::tan(fov / 2);
  m(0, 0) = 1 / (aspect * tan_half_fov);
  m(1, 1) = 1 / tan_half_fov;
  m(2, 2) = -(far + near) / (far - near);
  m(2, 3) = -2 * far * near / (far - near);
  m(3, 2) = -1;
  return m;
}

/**
 * @brief Normalize a vector
 * @tparam T Arithmetic type of matrix elements.
 * @tparam N Number of rows in the matrix.
 * @param v Vector to normalize.
 * @return Normalized vector.
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, 1> normalize(const Matrix<T, N, 1>& v) {
  T length = norm(v);
  if (std::abs(length) < std::numeric_limits<T>::epsilon()) {
    return v;
  }
  Matrix<T, N, 1> result;
  for (size_t i = 0; i < N; ++i) {
    result(i, 0) = v(i, 0) / length;
  }
  return result;
}

// Specialization for vec3
inline core::math::vector::Vector3 normalize(
    const core::math::vector::Vector3& v) {
  float length = v.magnitude();
  if (std::abs(length) < std::numeric_limits<float>::epsilon()) {
    return v;
  }
  return core::math::vector::Vector3{v[0] / length, v[1] / length,
                                     v[2] / length};
}
}  // namespace transform

/**
 * @namespace parallel
 * @brief Parallel matrix operations
 */
namespace parallel {
/**
 * @brief Parallel matrix addition
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the matrix.
 * @tparam Cols Number of columns in the matrix.
 * @param result Result matrix.
 * @param a First matrix.
 * @param b Second matrix.
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
void parallel_matrix_add(Matrix<T, Rows, Cols>& result,
                         const Matrix<T, Rows, Cols>& a,
                         const Matrix<T, Rows, Cols>& b) {
  if constexpr (Rows * Cols < 1000) {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = a(i, j) + b(i, j);
      }
    }
  } else {
    core::math::parallel::parallel_for(size_t(0), Rows, [&](size_t i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = a(i, j) + b(i, j);
      }
    });
  }
}

/**
 * @brief Parallel matrix multiplication
 * @tparam T Arithmetic type of matrix elements.
 * @tparam Rows Number of rows in the result matrix.
 * @tparam Inner Number of columns in the first matrix and rows in the second
 * matrix.
 * @tparam Cols Number of columns in the result matrix.
 * @param result Result matrix.
 * @param a First matrix.
 * @param b Second matrix.
 * @param block_size Size of the blocks.
 */
template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
void parallel_matrix_multiply(Matrix<T, Rows, Cols>& result,
                              const Matrix<T, Rows, Inner>& a,
                              const Matrix<T, Inner, Cols>& b,
                              size_t block_size = 64) {
  if constexpr (Rows * Cols * Inner < 1000) {
    block_multiply(result, a, b, block_size);
  } else {
    core::math::parallel::parallel_for(
        size_t(0), Rows / block_size + 1, [&](size_t bi) {
          size_t i_start = bi * block_size;
          size_t i_end = std::min(i_start + block_size, Rows);
          for (size_t j = 0; j < Cols; j += block_size) {
            size_t j_end = std::min(j + block_size, Cols);
            for (size_t k = 0; k < Inner; k += block_size) {
              size_t k_end = std::min(k + block_size, Inner);
              for (size_t ii = i_start; ii < i_end; ++ii) {
                for (size_t jj = j; jj < j_end; ++jj) {
                  T sum = result(ii, jj);
                  for (size_t kk = k; kk < k_end; ++kk) {
                    sum += a(ii, kk) * b(kk, jj);
                  }
                  result(ii, jj) = sum;
                }
              }
            }
          }
        });
  }
}
}  // namespace parallel

/**
 * @brief Sparse matrix in Compressed Sparse Row (CSR) format
 * @tparam T Numeric type satisfying ArithmeticValue concept
 */
template <ArithmeticValue T>
class SparseMatrix {
 private:
  std::vector<T, core::math::alloc::AlignedAllocator<T>>
      values_;                       // Non-zero values
  std::vector<size_t> col_indices_;  // Column indices of non-zeros
  std::vector<size_t> row_ptr_;      // Start of each row (size: rows_ + 1)
  size_t rows_, cols_;               // Matrix dimensions
  size_t nnz_;                       // Number of non-zero elements

 public:
  /**
   * @brief Constructor for SparseMatrix.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  SparseMatrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), nnz_(0) {
    row_ptr_.resize(rows + 1, 0);
  }

  /**
   * @brief Constructor for SparseMatrix from triplets.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param triplets Vector of triplets (row, column, value).
   */
  SparseMatrix(size_t rows, size_t cols,
               const std::vector<std::tuple<size_t, size_t, T>>& triplets)
      : rows_(rows), cols_(cols), nnz_(0) {
    row_ptr_.resize(rows + 1, 0);
    from_triplets(triplets);
  }

  /**
   * @brief Fill the matrix from triplets.
   * @param triplets Vector of triplets (row, column, value).
   */
  void from_triplets(
      const std::vector<std::tuple<size_t, size_t, T>>& triplets) {
    values_.clear();
    col_indices_.clear();
    row_ptr_.assign(rows_ + 1, 0);
    nnz_ = 0;

    for (const auto& [row, col, val] : triplets) {
      if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Triplet index out of bounds.");
      }
      if (std::abs(val) > std::numeric_limits<T>::epsilon()) {
        row_ptr_[row + 1]++;
        nnz_++;
      }
    }

    for (size_t i = 1; i <= rows_; ++i) {
      row_ptr_[i] += row_ptr_[i - 1];
    }

    std::vector<std::tuple<size_t, size_t, T>> sorted_triplets = triplets;
    std::sort(sorted_triplets.begin(), sorted_triplets.end(),
              [](const auto& a, const auto& b) {
                return std::tie(std::get<0>(a), std::get<1>(a)) <
                       std::tie(std::get<0>(b), std::get<1>(b));
              });

    values_.reserve(nnz_);
    col_indices_.reserve(nnz_);
    std::vector<size_t> current_pos(rows_, 0);

    size_t last_row = rows_, last_col = cols_;
    T accumulated = T(0);
    for (const auto& [row, col, val] : sorted_triplets) {
      if (std::abs(val) <= std::numeric_limits<T>::epsilon()) continue;

      if (row == last_row && col == last_col) {
        accumulated += val;
      } else {
        if (last_row < rows_ &&
            std::abs(accumulated) > std::numeric_limits<T>::epsilon()) {
          size_t pos = row_ptr_[last_row] + current_pos[last_row]++;
          values_.push_back(accumulated);
          col_indices_.push_back(last_col);
        }
        accumulated = val;
        last_row = row;
        last_col = col;
      }
    }
    if (last_row < rows_ &&
        std::abs(accumulated) > std::numeric_limits<T>::epsilon()) {
      size_t pos = row_ptr_[last_row] + current_pos[last_row]++;
      values_.push_back(accumulated);
      col_indices_.push_back(last_col);
    }

    for (size_t i = 0; i < rows_; ++i) {
      row_ptr_[i + 1] = row_ptr_[i] + current_pos[i];
    }
    nnz_ = values_.size();
  }

  /**
   * @brief Insert a value into the matrix.
   * @param row Row index.
   * @param col Column index.
   * @param value Value to insert.
   */
  void insert(size_t row, size_t col, T value) {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Index out of bounds.");
    }
    if (std::abs(value) <= std::numeric_limits<T>::epsilon()) {
      return;
    }

    std::vector<std::tuple<size_t, size_t, T>> triplets;
    triplets.reserve(nnz_ + 1);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
        triplets.emplace_back(i, col_indices_[j], values_[j]);
      }
    }
    triplets.emplace_back(row, col, value);
    from_triplets(triplets);
  }

  /**
   * @brief Access matrix element (const)
   * @param row Row index.
   * @param col Column index.
   * @return Value at the specified position.
   */
  T operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("Index out of bounds.");
    }
    for (size_t i = row_ptr_[row]; i < row_ptr_[i + 1]; ++i) {
      if (col_indices_[i] == col) {
        return values_[i];
      }
    }
    return T(0);
  }

  /**
   * @brief Multiply the matrix with a vector
   * @param vec Vector to multiply with.
   * @return Result of the multiplication.
   */
  std::vector<T> multiply(const Matrix<T, 4, 1>& vec) const {
    if (4 != cols_) {
      throw std::invalid_argument("Vector size mismatch: expected " +
                                  std::to_string(cols_) + " rows, got 4");
    }
    std::vector<T> result(rows_, T(0));
    if (nnz_ * rows_ < 1000) {
      for (size_t i = 0; i < rows_; ++i) {
        T sum = T(0);
        for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
          sum += values_[j] * vec(col_indices_[j], 0);
        }
        result[i] = sum;
      }
    } else {
      core::math::parallel::parallel_for(size_t(0), rows_, [&](size_t i) {
        T sum = T(0);
        for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
          sum += values_[j] * vec(col_indices_[j], 0);
        }
        result[i] = sum;
      });
    }
    return result;
  }

  /**
   * @brief Multiply the matrix with another sparse matrix
   * @param other Matrix to multiply with.
   * @return Result of the multiplication.
   */
  SparseMatrix<T> multiply(const SparseMatrix<T>& other) const {
    if (cols_ != other.rows_) {
      throw std::invalid_argument("Matrix dimension mismatch.");
    }
    SparseMatrix<T> result(rows_, other.cols_);
    std::vector<std::tuple<size_t, size_t, T>> triplets;
    triplets.reserve(nnz_ * other.nnz_ / rows_);

    for (size_t i = 0; i < rows_; ++i) {
      std::vector<T> row_accum(other.cols_, T(0));
      for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
        size_t k = col_indices_[j];
        T val = values_[j];
        for (size_t m = other.row_ptr_[k]; m < other.row_ptr_[k + 1]; ++m) {
          row_accum[other.col_indices_[m]] += val * other.values_[m];
        }
      }
      for (size_t j = 0; j < other.cols_; ++j) {
        if (std::abs(row_accum[j]) > std::numeric_limits<T>::epsilon()) {
          triplets.emplace_back(i, j, row_accum[j]);
        }
      }
    }
    result.from_triplets(triplets);
    return result;
  }

  /**
   * @brief Add two sparse matrices
   * @param other Matrix to add.
   * @return Result of the addition.
   */
  SparseMatrix<T> operator+(const SparseMatrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      throw std::invalid_argument("Matrix dimension mismatch.");
    }
    std::vector<std::tuple<size_t, size_t, T>> triplets;
    triplets.reserve(nnz_ + other.nnz_);
    for (size_t i = 0; i < rows_; ++i) {
      size_t j1 = row_ptr_[i], j2 = other.row_ptr_[i];
      while (j1 < row_ptr_[i + 1] || j2 < other.row_ptr_[i + 1]) {
        size_t col;
        T val = T(0);
        if (j1 < row_ptr_[i + 1] &&
            (j2 >= other.row_ptr_[i + 1] ||
             col_indices_[j1] < other.col_indices_[j2])) {
          col = col_indices_[j1];
          val = values_[j1];
          ++j1;
        } else if (j2 < other.row_ptr_[i + 1] &&
                   (j1 >= row_ptr_[i + 1] ||
                    other.col_indices_[j2] < col_indices_[j1])) {
          col = other.col_indices_[j2];
          val = other.values_[j2];
          ++j2;
        } else {
          col = col_indices_[j1];
          val = values_[j1] + other.values_[j2];
          ++j1;
          ++j2;
        }
        if (std::abs(val) > std::numeric_limits<T>::epsilon()) {
          triplets.emplace_back(i, col, val);
        }
      }
    }
    return SparseMatrix<T>(rows_, cols_, triplets);
  }

  /**
   * @brief Transpose the matrix
   * @return Transposed matrix.
   */
  SparseMatrix<T> transpose() const {
    SparseMatrix<T> result(cols_, rows_);
    std::vector<size_t> count(cols_, 0);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
        count[col_indices_[j]]++;
      }
    }
    result.row_ptr_.resize(cols_ + 1);
    result.row_ptr_[0] = 0;
    for (size_t i = 0; i < cols_; ++i) {
      result.row_ptr_[i + 1] = result.row_ptr_[i] + count[i];
    }
    result.values_.resize(nnz_);
    result.col_indices_.resize(nnz_);
    count.assign(cols_, 0);
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
        size_t new_row = col_indices_[j];
        size_t pos = result.row_ptr_[new_row] + count[new_row]++;
        result.values_[pos] = values_[j];
        result.col_indices_[pos] = i;
      }
    }
    result.nnz_ = nnz_;
    return result;
  }

  /**
   * @brief Compute the Frobenius norm of the matrix
   * @return Frobenius norm.
   */
  T frobenius_norm() const {
    T sum = T(0);
    if (nnz_ < 1000) {
      for (const auto& val : values_) {
        sum += val * val;
      }
    } else {
      T partial_sum = T(0);
      core::math::parallel::parallel_for(size_t(0), values_.size(),
                                         [&](size_t i) {
                                           T val = values_[i];
                                           partial_sum += val * val;
                                         });
      sum = partial_sum;
    }
    return std::sqrt(sum);
  }

  /**
   * @brief Get the number of non-zero elements
   * @return Number of non-zero elements.
   */
  size_t non_zeros() const { return nnz_; }

  /**
   * @brief Compute the sparsity ratio of the matrix
   * @return Sparsity ratio.
   */
  double sparsity_ratio() const {
    return static_cast<double>(nnz_) / (rows_ * cols_);
  }

  /**
   * @brief Get the dimensions of the matrix
   * @return Pair of rows and columns.
   */
  std::pair<size_t, size_t> size() const { return {rows_, cols_}; }

  /**
   * @brief Iterator for SparseMatrix
   */
  class Iterator {
   private:
    const SparseMatrix<T>* matrix_;
    size_t pos_;

   public:
    /**
     * @brief Constructor for Iterator.
     * @param matrix Pointer to the matrix.
     * @param pos Position.
     */
    Iterator(const SparseMatrix<T>* matrix, size_t pos)
        : matrix_(matrix), pos_(pos) {}

    /**
     * @brief Check if two iterators are not equal
     * @param other Other iterator.
     * @return True if not equal, false otherwise.
     */
    bool operator!=(const Iterator& other) const { return pos_ != other.pos_; }

    /**
     * @brief Increment the iterator
     * @return Reference to the iterator.
     */
    Iterator& operator++() {
      ++pos_;
      return *this;
    }

    /**
     * @brief Dereference the iterator
     * @return Tuple of row, column, and value.
     */
    std::tuple<size_t, size_t, T> operator*() const {
      size_t row = 0;
      while (row < matrix_->rows_ && matrix_->row_ptr_[row + 1] <= pos_) {
        ++row;
      }
      return {row, matrix_->col_indices_[pos_], matrix_->values_[pos_]};
    }
  };

  /**
   * @brief Get an iterator to the beginning of the matrix
   * @return Iterator to the beginning.
   */
  Iterator begin() const { return Iterator(this, 0); }

  /**
   * @brief Get an iterator to the end of the matrix
   * @return Iterator to the end.
   */
  Iterator end() const { return Iterator(this, nnz_); }
};

// Common matrix type aliases
using mat2x2 = Matrix<float, 2, 2>;
using mat2x3 = Matrix<float, 2, 3>;
using mat2x4 = Matrix<float, 2, 4>;
using mat3x2 = Matrix<float, 3, 2>;
using mat3x3 = Matrix<float, 3, 3>;
using mat3x4 = Matrix<float, 3, 4>;
using mat4x2 = Matrix<float, 4, 2>;
using mat4x3 = Matrix<float, 4, 3>;
using mat4x4 = Matrix<float, 4, 4>;

/**
 * @brief Multiply two 4x4 matrices with unrolled loops
 * @param a First matrix.
 * @param b Second matrix.
 * @return Result of the multiplication.
 */
mat4x4 multiply_unrolled(const mat4x4& a, const mat4x4& b) {
  mat4x4 result;
  result(0, 0) = a(0, 0) * b(0, 0) + a(0, 1) * b(1, 0) + a(0, 2) * b(2, 0) +
                 a(0, 3) * b(3, 0);
  result(0, 1) = a(0, 0) * b(0, 1) + a(0, 1) * b(1, 1) + a(0, 2) * b(2, 1) +
                 a(0, 3) * b(3, 1);
  // и т.д. для всех элементов
  return result;
}

/**
 * @brief Create a view matrix for a camera
 * @param eye Position of the camera.
 * @param target Target position.
 * @param up Up vector.
 * @return View matrix.
 */
mat4x4 look_at(const core::math::vector::Vector3& eye,
               const core::math::vector::Vector3& target,
               const core::math::vector::Vector3& up) {
  using namespace core::math::vector;
  // Compute the forward (z), right (x), and up (y) vectors
  Vector3 z = (target - eye).normalize();  // Vector library's normalize
  Vector3 x = up.cross(z).normalize();  // Vector library's cross and normalize
  Vector3 y = z.cross(x);               // Vector library's cross

  // Construct the view matrix
  mat4x4 view = mat4x4::identity();
  view(0, 0) = x[0];
  view(0, 1) = x[1];
  view(0, 2) = x[2];
  view(0, 3) = -x.dot(eye);  // Vector library's dot
  view(1, 0) = y[0];
  view(1, 1) = y[1];
  view(1, 2) = y[2];
  view(1, 3) = -y.dot(eye);  // Vector library's dot
  view(2, 0) = z[0];
  view(2, 1) = z[1];
  view(2, 2) = z[2];
  view(2, 3) = -z.dot(eye);  // Vector library's dot
  view(3, 3) = 1.0f;

  return view;
}

}  // namespace core::math::matrix
/** @} */  // end of core_math_matrix group
