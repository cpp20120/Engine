#pragma once

#include <array>
#include <concepts>
#include <initializer_list>
#include <mdspan/mdspan.hpp>
#include <mdspan>
#include <print>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

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
concept ArithmeticValue = requires([[maybe_unused]] T a, [[maybe_unused]] T b) {
  requires std::is_arithmetic_v<T>;
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
  { a / b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
};

/**
 * @brief Compile-time matrix implementation with various linear algebra
 * operations
 *
 * @tparam T Numeric type for matrix elements
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 *
 * Example usage:
 * @code
 * Matrix<double, 3, 3> mat = {
 *   1.0, 2.0, 3.0,
 *   4.0, 5.0, 6.0,
 *   7.0, 8.0, 9.0
 * };
 * @endcode
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
class Matrix {
 private:
  std::vector<T> data_;  ///< Vector with data of matrix
  std::mdspan<T, std::extents<size_t, Rows, Cols>>
      view_;  ///< Multidimensional view above data

 public:
  /**
   * @brief Default constructor initializes matrix to zeros
   */
  constexpr Matrix() : data_(Rows * Cols), view_(data_.data()) {}

  /**
   * @brief Construct from initializer list
   * @param init Initializer list with Rows*Cols elements
   * @throws std::invalid_argument if size doesn't match matrix dimensions
   */
  constexpr Matrix(std::initializer_list<T> init)
      : data_(init), view_(data_.data()) {
    if (init.size() != Rows * Cols) {
      throw std::invalid_argument(
          "Initializer list size does not match matrix dimensions.");
    }
  }

  /**
   * @brief Move constructor for Matrix
   * @param other Matrix to move from
   */
  Matrix(Matrix&& other) noexcept
      : data_(std::move(other.data_)), view_(data_.data()) {
    other.data_.clear();
    other.data_.shrink_to_fit();
    other.view_ =
        std::mdspan<T, std::extents<size_t, Rows, Cols>>(other.data_.data());
  }

  /**
   * @brief Get mutable view of the matrix
   * @return mdspan view of the matrix data
   */
  constexpr std::mdspan<T, std::extents<size_t, Rows, Cols>, std::layout_right>
  get_view() {
    return view_;
  }

  /**
   * @brief Get const view of the matrix
   * @return const mdspan view of the matrix data
   */
  constexpr auto get_view() const { return view_; }

  /**
   * @brief Access matrix element (mutable)
   * @param i Row index (0-based)
   * @param j Column index (0-based)
   * @return Reference to matrix element
   */
  constexpr T& operator()(size_t i, size_t j) { return view_[i, j]; }

  /**
   * @brief Access matrix element (const)
   * @param i Row index (0-based)
   * @param j Column index (0-based)
   * @return Const reference to matrix element
   */
  constexpr const T& operator()(size_t i, size_t j) const {
    return view_[i, j];
  }

  /**
   * @brief Get pointer to underlying data
   * @return Pointer to contiguous matrix data
   */
  constexpr T* data() noexcept { return data_.data(); }

  /**
   * @brief Get const pointer to underlying data
   * @return Const pointer to contiguous matrix data
   */
  constexpr const T* data() const noexcept { return data_.data(); }

  /**
   * @brief Get matrix dimensions
   * @return Pair of (rows, columns)
   */
  constexpr std::pair<size_t, size_t> size() const noexcept {
    return {Rows, Cols};
  }

  /**
   * @brief Check if matrix is square
   * @return true if matrix is square, false otherwise
   */
  constexpr bool is_square() const noexcept { return Rows == Cols; }

  /**
   * @brief Print matrix to standard output
   */
  constexpr void print() const {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        std::print("{} ", view_[i, j]);
      }
      std::print("\n");
    }
  }

  /**
   * @brief Create identity matrix
   * @return Identity matrix of same dimensions
   * @note Only available for square matrices
   */
  static constexpr Matrix identity() {
    static_assert(Rows == Cols, "Identity matrix must be square.");
    Matrix result;
    for (size_t i = 0; i < Rows; ++i) {
      result(i, i) = 1;
    }
    return result;
  }

  /**
   * @brief Set all matrix elements to zero
   */
  constexpr void zero() {
    for (auto& val : data_) {
      val = 0;
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
   * @brief Iterator to beginning of elements via mdspan
   * @return Iterator to first element
   */
  constexpr auto elements_begin() noexcept { return view_.data(); }

  /**
   * @brief Iterator to end of elements via mdspan
   * @return Iterator to element after last
   */
  constexpr auto elements_end() noexcept { return view_.data() + view_.size(); }

  /**
   * @brief Const iterator to beginning of elements via mdspan
   * @return Const iterator to first element
   */
  constexpr auto elements_begin() const noexcept { return view_.data(); }

  /**
   * @brief Const iterator to end of elements via mdspan
   * @return Const iterator to element after last
   */
  constexpr auto elements_end() const noexcept {
    return view_.data() + view_.size();
  }

  /**
   * @brief Create a submatrix slice
   * @tparam StartRow Starting row index
   * @tparam EndRow Ending row index (exclusive)
   * @tparam StartCol Starting column index
   * @tparam EndCol Ending column index (exclusive)
   * @return Submatrix view
   *
   * Example usage:
   * @code
   * auto sub = matrix.slice<1, 3, 0, 2>();
   * @endcode
   */
  template <size_t StartRow, size_t EndRow, size_t StartCol, size_t EndCol>
  auto slice() {
    static_assert(StartRow < EndRow && EndRow <= Rows);
    static_assert(StartCol < EndCol && EndCol <= Cols);

    return std::submdspan(view_, std::tuple{StartRow, EndRow},
                          std::tuple{StartCol, EndCol});
  }

  /**
   * @brief Matrix addition
   * @param other Matrix to add
   * @return Resulting matrix
   */
  constexpr Matrix operator+(const Matrix& other) const {
    Matrix result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Matrix subtraction
   * @param other Matrix to subtract
   * @return Resulting matrix
   */
  constexpr Matrix operator-(const Matrix& other) const {
    Matrix result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = (*this)(i, j) - other(i, j);
      }
    }
    return result;
  }

  /**
   * @brief Scalar multiplication
   * @param scalar Scalar value to multiply by
   * @return Resulting matrix
   */
  constexpr Matrix operator*(T scalar) const {
    Matrix result;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) = (*this)(i, j) * scalar;
      }
    }
    return result;
  }

  /**
   * @brief Matrix equality comparison
   * @param other Matrix to compare with
   * @return true if all elements are equal, false otherwise
   */
  constexpr bool operator==(const Matrix& other) const {
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        if ((*this)(i, j) != other(i, j)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Matrix inequality comparison
   * @param other Matrix to compare with
   * @return true if any elements differ, false otherwise
   */
  constexpr bool operator!=(const Matrix& other) const {
    return !(*this == other);
  }

  /**
   * @brief Move assignment operator
   * @param other Matrix to move from
   * @return Reference to this matrix after move
   */
  Matrix& operator=(Matrix&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = std::mdspan<T, std::extents<size_t, Rows, Cols>>(data_.data());
      other.data_.clear();
      other.data_.shrink_to_fit();
      other.view_ =
          std::mdspan<T, std::extents<size_t, Rows, Cols>>(other.data_.data());
    }
    return *this;
  }
};

/**
 * @brief Matrix addition
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param a First matrix
 * @param b Second matrix
 * @return Resulting matrix
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> add(const Matrix<T, Rows, Cols>& a,
                                    const Matrix<T, Rows, Cols>& b) {
  Matrix<T, Rows, Cols> result;
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }
  return result;
}

/**
 * @brief Matrix subtraction
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param a First matrix
 * @param b Second matrix
 * @return Resulting matrix
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
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Input matrix
 * @param scalar Scalar value
 * @return Resulting matrix
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
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Input matrix
 * @return Transposed matrix
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
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @return Determinant value
 */
template <ArithmeticValue T, size_t N>
constexpr T determinant(const Matrix<T, N, N>& matrix) {
  if constexpr (N == 1) {
    return matrix(0, 0);
  } else {
    T det = 0;
    for (size_t i = 0; i < N; ++i) {
      Matrix<T, N - 1, N - 1> submatrix;
      for (size_t row = 1; row < N; ++row) {
        size_t colIndex = 0;
        for (size_t col = 0; col < N; ++col) {
          if (col == i) continue;
          submatrix(row - 1, colIndex) = matrix(row, col);
          ++colIndex;
        }
      }
      det += (i % 2 == 0 ? 1 : -1) * matrix(0, i) * determinant(submatrix);
    }
    return det;
  }
}

/**
 * @brief Compute determinant via LU decomposition
 * @tparam T Numeric type
 * @tparam N Matrix dimension
 * @param matrix Input matrix
 * @return Determinant value
 * @note More numerically stable for larger matrices than recursive approach
 */
template <typename T, size_t N>
T determinant_via_lu(const Matrix<T, N, N>& matrix) {
  auto [L, U, P] = lu_decomposition(matrix);
  T det = 1;
  for (size_t i = 0; i < N; ++i) {
    det *= U(i, i);  /// Произведение диагональных элементов U
  }
  // Учитываем перестановки в матрице P
  size_t num_swaps = 0;
  for (size_t i = 0; i < N; ++i) {
    if (P(i, i) != 1) {
      num_swaps++;
    }
  }
  if (num_swaps % 2 != 0) {
    det = -det;  // Корректируем знак определителя
  }
  return det;
}

/// Specialization for 1x1 matrix
template <typename T>
constexpr T determinant(const Matrix<T, 1, 1>& matrix) {
  return matrix(0, 0);
}

/// Specialization for 2x2 matrix
template <typename T>
constexpr T determinant(const Matrix<T, 2, 2>& matrix) {
  return matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0);
}

/// Specialization for 3x3 matrix (rule of Sarrus)
template <typename T>
constexpr T determinant(const Matrix<T, 3, 3>& matrix) {
  return matrix(0, 0) * matrix(1, 1) * matrix(2, 2) +
         matrix(0, 1) * matrix(1, 2) * matrix(2, 0) +
         matrix(0, 2) * matrix(1, 0) * matrix(2, 1) -
         matrix(0, 2) * matrix(1, 1) * matrix(2, 0) -
         matrix(0, 0) * matrix(1, 2) * matrix(2, 1) -
         matrix(0, 1) * matrix(1, 0) * matrix(2, 2);
}

/// For NxN matrix
template <typename T, size_t N>
T determinant(const Matrix<T, N, N>& matrix) {
  if constexpr (N == 1) {
    return determinant(matrix);
  } else if constexpr (N == 2) {
    return determinant(matrix);
  } else if constexpr (N == 3) {
    return determinant(matrix);
  } else {
    return determinant_via_lu(matrix);
  }
}

/**
 * @brief Matrix multiplication
 * @tparam T Numeric type
 * @tparam Rows Rows in first matrix
 * @tparam Inner Inner dimension
 * @tparam Cols Columns in second matrix
 * @param a First matrix
 * @param b Second matrix
 * @return Product matrix
 */
template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
constexpr Matrix<T, Rows, Cols> multiply(const Matrix<T, Rows, Inner>& a,
                                         const Matrix<T, Inner, Cols>& b) {
  Matrix<T, Rows, Cols> result;
  result.zero();
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t k = 0; k < Inner; ++k) {
      for (size_t j = 0; j < Cols; ++j) {
        result(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return result;
}

/**
 * @brief Matrix inverse
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @return Inverse matrix
 * @throws std::runtime_error if matrix is singular
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, N> inverse(const Matrix<T, N, N>& matrix) {
  Matrix<T, N, N> identity = Matrix<T, N, N>::identity();
  Matrix<T, N, N> copy = matrix;

  for (size_t i = 0; i < N; ++i) {
    T diag = copy(i, i);
    if (std::abs(diag) < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }
    for (size_t j = 0; j < N; ++j) {
      copy(i, j) /= diag;
      identity(i, j) /= diag;
    }
    for (size_t k = 0; k < N; ++k) {
      if (k != i) {
        T factor = copy(k, i);
        for (size_t j = 0; j < N; ++j) {
          copy(k, j) -= factor * copy(i, j);
          identity(k, j) -= factor * identity(i, j);
        }
      }
    }
  }
  return identity;
}

/// Operator overloads for matrix operations

/**
 * @brief Matrix addition operator
 * @copydetails add()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& a,
                                          const Matrix<T, Rows, Cols>& b) {
  return add(a, b);
}

/**
 * @brief Matrix subtraction operator
 * @copydetails subtract()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& a,
                                          const Matrix<T, Rows, Cols>& b) {
  return subtract(a, b);
}

/**
 * @brief Scalar multiplication operator (matrix * scalar)
 * @copydetails scalar_multiply()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, Cols>& matrix,
                                          T scalar) {
  return scalar_multiply(matrix, scalar);
}

/**
 * @brief Scalar multiplication operator (scalar * matrix)
 * @copydetails scalar_multiply()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(T scalar,
                                          const Matrix<T, Rows, Cols>& matrix) {
  return matrix * scalar;
}

/**
 * @brief Matrix multiplication operator
 * @copydetails multiply()
 */
template <ArithmeticValue T, size_t Rows, size_t Inner, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(const Matrix<T, Rows, Inner>& a,
                                          const Matrix<T, Inner, Cols>& b) {
  return multiply(a, b);
}

/**
 * @brief Matrix equality operator
 * @copydetails Matrix::operator==()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr bool operator==(const Matrix<T, Rows, Cols>& a,
                          const Matrix<T, Rows, Cols>& b) {
  return a.operator==(b);
}

/**
 * @brief Matrix inequality operator
 * @copydetails Matrix::operator!=()
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr bool operator!=(const Matrix<T, Rows, Cols>& a,
                          const Matrix<T, Rows, Cols>& b) {
  return !(a == b);
}

/**
 * @brief Matrix trace (sum of diagonal elements)
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @return Trace value
 */
template <ArithmeticValue T, size_t N>
constexpr T trace(const Matrix<T, N, N>& matrix) {
  T sum = 0;
  for (size_t i = 0; i < N; ++i) {
    sum += matrix(i, i);
  }
  return sum;
}

/**
 * @brief Matrix minor (submatrix excluding row and column)
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @param row Row to exclude
 * @param col Column to exclude
 * @return Minor matrix
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
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @param row Row index
 * @param col Column index
 * @return Cofactor value
 */
template <ArithmeticValue T, size_t N>
constexpr T cofactor(const Matrix<T, N, N>& matrix, size_t row, size_t col) {
  return ((row + col) % 2 == 0 ? 1 : -1) * determinant(minor(matrix, row, col));
}

/**
 * @brief Cofactor matrix
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @return Cofactor matrix
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
 * @brief Matrix adjugate (transpose of cofactor matrix)
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param matrix Input matrix
 * @return Adjugate matrix
 */
template <ArithmeticValue T, size_t N>
constexpr Matrix<T, N, N> adjugate(const Matrix<T, N, N>& matrix) {
  return transpose(cofactor_matrix(matrix));
}

/**
 * @brief Matrix rank
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Input matrix
 * @return Matrix rank
 */
template <ArithmeticValue T, size_t Rows, size_t Cols>
constexpr size_t rank(Matrix<T, Rows, Cols> matrix) {
  size_t rank = 0;
  for (size_t row = 0; row < Rows; ++row) {
    size_t leading_col = row;
    while (leading_col < Cols && matrix(row, leading_col) == 0) {
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
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Input matrix
 * @param r Row index
 * @return Array containing row elements
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
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Input matrix
 * @param c Column index
 * @return Array containing column elements
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
 * @tparam T Numeric type
 * @tparam Rows Number of rows
 * @tparam Cols Number of columns
 * @param matrix Matrix to fill
 * @param min Minimum random value
 * @param max Maximum random value
 */
// Primary template for randomizing with min/max range
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

// Overload for default range [0,1]
template <typename T, size_t Rows, size_t Cols>
void randomize(Matrix<T, Rows, Cols>& matrix) {
  randomize(matrix, static_cast<T>(0), static_cast<T>(1));
}

// Overload for symmetric range [-range, range]
template <typename T, size_t Rows, size_t Cols>
void randomize(Matrix<T, Rows, Cols>& matrix, T range) {
  randomize(matrix, -range, range);
}

// Overload with custom generator
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
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param A Input matrix
 * @return Pair of Q (orthogonal) and R (upper triangular) matrices
 */
template <ArithmeticValue T, size_t N>
constexpr std::pair<Matrix<T, N, N>, Matrix<T, N, N>> qr_decomposition(
    const Matrix<T, N, N>& A) {
  Matrix<T, N, N> Q = Matrix<T, N, N>::identity();
  Matrix<T, N, N> R = A;

  for (size_t k = 0; k < N - 1; ++k) {
    T norm = 0;
    for (size_t i = k; i < N; ++i) {
      norm += R(i, k) * R(i, k);
    }
    norm = std::sqrt(norm);

    T alpha = -std::copysign(norm, R(k, k));
    T r = std::sqrt(0.5 * (alpha * alpha - R(k, k) * alpha));

    std::array<T, N> v{};
    v[k] = (R(k, k) - alpha) / (2 * r);
    for (size_t i = k + 1; i < N; ++i) {
      v[i] = R(i, k) / (2 * r);
    }

    for (size_t j = k; j < N; ++j) {
      T dot = 0;
      for (size_t i = k; i < N; ++i) {
        dot += v[i] * R(i, j);
      }
      for (size_t i = k; i < N; ++i) {
        R(i, j) -= 2 * v[i] * dot;
      }
    }

    // Apply Householder transformation to Q
    for (size_t j = 0; j < N; ++j) {
      T dot = 0;
      for (size_t i = k; i < N; ++i) {
        dot += v[i] * Q(j, i);
      }
      for (size_t i = k; i < N; ++i) {
        Q(j, i) -= 2 * v[i] * dot;
      }
    }
  }

  return {transpose(Q), R};
}

/**
 * @brief LU decomposition using Doolittle's method
 * @tparam T Numeric type
 * @tparam N Matrix dimension (square)
 * @param A Input matrix
 * @return Tuple of L (lower triangular), U (upper triangular), and P
 * (permutation) matrices
 */
template <ArithmeticValue T, size_t N>
constexpr std::tuple<Matrix<T, N, N>, Matrix<T, N, N>, Matrix<T, N, N>>
lu_decomposition(const Matrix<T, N, N>& A) {
  Matrix<T, N, N> L = Matrix<T, N, N>::identity();
  Matrix<T, N, N> U{};
  Matrix<T, N, N> P = Matrix<T, N, N>::identity();

  for (size_t k = 0; k < N; ++k) {
    size_t max_row = k;
    T max_val = std::abs(A(k, k));
    for (size_t i = k + 1; i < N; ++i) {
      if (std::abs(A(i, k)) > max_val) {
        max_val = std::abs(A(i, k));
        max_row = i;
      }
    }

    if (max_row != k) {
      for (size_t j = 0; j < N; ++j) {
        std::swap(P(k, j), P(max_row, j));
        std::swap(L(k, j), L(max_row, j));
        std::swap(U(k, j), U(max_row, j));
      }
    }

    for (size_t j = k; j < N; ++j) {
      U(k, j) = A(k, j);
      for (size_t m = 0; m < k; ++m) {
        U(k, j) -= L(k, m) * U(m, j);
      }
    }

    for (size_t i = k + 1; i < N; ++i) {
      L(i, k) = A(i, k);
      for (size_t m = 0; m < k; ++m) {
        L(i, k) -= L(i, m) * U(m, k);
      }
      L(i, k) /= U(k, k);
    }
  }

  return {L, U, P};
}

/**
 * @namespace transform
 * @brief Geometric transformation matrices
 */
namespace transform {

/**
 * @brief Create 2D translation matrix
 * @tparam T Numeric type
 * @param x X translation
 * @param y Y translation
 * @return 3x3 transformation matrix
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> translate2d(T x, T y) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  m(0, 2) = x;
  m(1, 2) = y;
  return m;
}

/**
 * @brief Create 2D rotation matrix
 * @tparam T Numeric type
 * @param angle_rad Rotation angle in radians
 * @return 3x3 transformation matrix
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
 * @brief Create 2D scaling matrix
 * @tparam T Numeric type
 * @param sx X scale factor
 * @param sy Y scale factor
 * @return 3x3 transformation matrix
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> scale2d(T sx, T sy) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  m(0, 0) = sx;
  m(1, 1) = sy;
  return m;
}

/**
 * @brief Create 2D reflection matrix
 * @tparam T Numeric type
 * @param x_axis Reflect across x-axis
 * @param y_axis Reflect across y-axis
 * @return 3x3 transformation matrix
 */
template <ArithmeticValue T>
constexpr Matrix<T, 3, 3> reflect2d(bool x_axis, bool y_axis) {
  Matrix<T, 3, 3> m = Matrix<T, 3, 3>::identity();
  if (x_axis) m(1, 1) = -1;
  if (y_axis) m(0, 0) = -1;
  return m;
}

/**
 * @brief Create 3D translation matrix
 * @tparam T Numeric type
 * @param x X translation
 * @param y Y translation
 * @param z Z translation
 * @return 4x4 transformation matrix
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
 * @brief Create 3D rotation matrix around X-axis
 * @tparam T Numeric type
 * @param angle_rad Rotation angle in radians
 * @return 4x4 transformation matrix
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
 * @brief Create 3D rotation matrix around Y-axis
 * @tparam T Numeric type
 * @param angle_rad Rotation angle in radians
 * @return 4x4 transformation matrix
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
 * @brief Create 3D rotation matrix around Z-axis
 * @tparam T Numeric type
 * @param angle_rad Rotation angle in radians
 * @return 4x4 transformation matrix
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
 * @brief Create 3D scaling matrix
 * @tparam T Numeric type
 * @param sx X scale factor
 * @param sy Y scale factor
 * @param sz Z scale factor
 * @return 4x4 transformation matrix
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
 * @brief Create 3D reflection matrix
 * @tparam T Numeric type
 * @param x_axis Reflect across x-axis
 * @param y_axis Reflect across y-axis
 * @param z_axis Reflect across z-axis
 * @return 4x4 transformation matrix
 */
template <ArithmeticValue T>
constexpr Matrix<T, 4, 4> reflect3d(bool x_axis, bool y_axis, bool z_axis) {
  Matrix<T, 4, 4> m = Matrix<T, 4, 4>::identity();
  if (x_axis) m(0, 0) = -1;
  if (y_axis) m(1, 1) = -1;
  if (z_axis) m(2, 2) = -1;
  return m;
}

}  // namespace transform

/// Common matrix type aliases
using mat2x2 = Matrix<float, 2, 2>;  ///< 2x2 float matrix
using mat2x3 = Matrix<float, 2, 3>;  ///< 2x3 float matrix
using mat2x4 = Matrix<float, 2, 4>;  ///< 2x4 float matrix
using mat3x2 = Matrix<float, 3, 2>;  ///< 3x2 float matrix
using mat3x3 = Matrix<float, 3, 3>;  ///< 3x3 float matrix
using mat3x4 = Matrix<float, 3, 4>;  ///< 3x4 float matrix
using mat4x2 = Matrix<float, 4, 2>;  ///< 4x2 float matrix
using mat4x3 = Matrix<float, 4, 3>;  ///< 4x3 float matrix
using mat4x4 = Matrix<float, 4, 4>;  ///< 4x4 float matrix
}  // namespace core::math::matrix
/** @} */  // end of core_math_matrix group