#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../matrix.hpp"  // Assuming this is the path to your matrix library

namespace core::math::matrix::api {

/**
 * @class MatrixAPI
 * @brief High-level matrix API with simplified interface.
 *
 * This class provides a high-level interface for matrix operations,
 * making it easier to perform common matrix operations without dealing
 * with the underlying complexity.
 */
class MatrixAPI {
 public:
  // Matrix Creation

  /**
   * @brief Creates a matrix of specified size.
   * @tparam T Type of matrix elements.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @return Matrix of specified size.
   * @throws std::runtime_error If the matrix size is unsupported.
   */
  template <typename T>
  static auto create(size_t rows, size_t cols) {
    if (rows == 2 && cols == 2) return Matrix2D();
    if (rows == 3 && cols == 3) return Matrix3D();
    if (rows == 4 && cols == 4) return Matrix4D();
    throw std::runtime_error("Unsupported matrix size");
  }

  /**
   * @brief Creates a matrix of specified template size.
   * @tparam T Type of matrix elements.
   * @tparam Rows Number of rows.
   * @tparam Cols Number of columns.
   * @return Matrix of specified template size.
   */
  template <typename T, size_t Rows, size_t Cols>
  static auto create() {
    return core::math::matrix::Matrix<T, Rows, Cols>();
  }

  /**
   * @brief Creates an identity matrix of specified size.
   * @tparam T Type of matrix elements.
   * @param size Size of the identity matrix.
   * @return Identity matrix of specified size.
   * @throws std::runtime_error If the identity matrix size is unsupported.
   */
  template <typename T>
  static auto identity(size_t size) {
    if (size == 2) return Matrix2D::identity();
    if (size == 3) return Matrix3D::identity();
    if (size == 4) return Matrix4D::identity();
    throw std::runtime_error("Unsupported identity matrix size");
  }

  /**
   * @brief Creates a matrix filled with zeros.
   * @tparam T Type of matrix elements.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @return Matrix filled with zeros.
   */
  template <typename T>
  static auto zeros(size_t rows, size_t cols) {
    auto mat = create<T>(rows, cols);
    mat.zero();
    return mat;
  }

  /**
   * @brief Creates a matrix filled with ones.
   * @tparam T Type of matrix elements.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @return Matrix filled with ones.
   */
  template <typename T>
  static auto ones(size_t rows, size_t cols) {
    auto mat = create<T>(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        mat(i, j) = T(1);
      }
    }
    return mat;
  }

  /**
   * @brief Creates a matrix filled with random values.
   * @tparam T Type of matrix elements.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param min Minimum value.
   * @param max Maximum value.
   * @return Matrix filled with random values.
   */
  template <typename T>
  static auto random(size_t rows, size_t cols, T min, T max) {
    auto mat = create<T>(rows, cols);
    core::math::matrix::randomize(mat, min, max);
    return mat;
  }

  // Basic Operations

  /**
   * @brief Adds two matrices.
   * @tparam T Type of matrix elements.
   * @param a First matrix.
   * @param b Second matrix.
   * @return Result of the addition.
   */
  template <typename T>
  static auto add(const T& a, const T& b) {
    return a + b;
  }

  /**
   * @brief Subtracts two matrices.
   * @tparam T Type of matrix elements.
   * @param a First matrix.
   * @param b Second matrix.
   * @return Result of the subtraction.
   */
  template <typename T>
  static auto subtract(const T& a, const T& b) {
    return a - b;
  }

  /**
   * @brief Multiplies two matrices.
   * @tparam T Type of matrix elements.
   * @param a First matrix.
   * @param b Second matrix.
   * @return Result of the multiplication.
   */
  template <typename T>
  static auto multiply(const T& a, const T& b) {
    return a * b;
  }

  /**
   * @brief Multiplies a matrix by a scalar.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to multiply.
   * @param scalar Scalar value.
   * @return Result of the scalar multiplication.
   */
  template <typename T>
  static auto scalar_multiply(const T& matrix, typename T::value_type scalar) {
    return matrix * scalar;
  }

  /**
   * @brief Transposes a matrix.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to transpose.
   * @return Transposed matrix.
   */
  template <typename T>
  static auto transpose(const T& matrix) {
    return core::math::matrix::transpose(matrix);
  }

  // Linear Algebra

  /**
   * @brief Computes the inverse of a matrix.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to invert.
   * @return Inverse of the matrix.
   */
  template <typename T>
  static auto inverse(const T& matrix) {
    return core::math::matrix::inverse(matrix);
  }

  /**
   * @brief Computes the determinant of a matrix.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to compute the determinant of.
   * @return Determinant of the matrix.
   */
  template <typename T>
  static auto determinant(const T& matrix) {
    return core::math::matrix::determinant(matrix);
  }

  /**
   * @brief Solves a linear system of equations.
   * @tparam T Type of matrix elements.
   * @param A Coefficient matrix.
   * @param b Right-hand side vector.
   * @return Solution vector.
   */
  template <typename T>
  static auto solve_linear_system(const T& A, const T& b) {
    return core::math::matrix::inverse(A) * b;
  }

  // Decompositions

  /**
   * @brief Performs LU decomposition of a matrix.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to decompose.
   * @return Tuple of L, U, and P matrices.
   */
  template <typename T>
  static auto lu_decomposition(const T& matrix) {
    return core::math::matrix::lu_decomposition(matrix);
  }

  /**
   * @brief Performs QR decomposition of a matrix.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to decompose.
   * @return Pair of Q and R matrices.
   */
  template <typename T>
  static auto qr_decomposition(const T& matrix) {
    return core::math::matrix::qr_decomposition(matrix);
  }

  // Transformations (3D Graphics)

  /**
   * @brief Creates a translation matrix.
   * @param x Translation in x direction.
   * @param y Translation in y direction.
   * @param z Translation in z direction.
   * @return Translation matrix.
   */
  static auto create_translation(float x, float y, float z = 0.0f) {
    if (z == 0.0f) {
      return core::math::matrix::transform::translate2d(x, y);
    }
    return core::math::matrix::transform::translate3d(x, y, z);
  }

  /**
   * @brief Creates a rotation matrix.
   * @param angle Rotation angle in radians.
   * @param x X component of the rotation axis.
   * @param y Y component of the rotation axis.
   * @param z Z component of the rotation axis.
   * @return Rotation matrix.
   * @throws std::runtime_error If the rotation axis is not axis-aligned.
   */
  static auto create_rotation(float angle, float x, float y, float z) {
    if (x == 0.0f && y == 0.0f && z == 1.0f) {
      return core::math::matrix::transform::rotate2d(angle);
    } else if (x == 1.0f && y == 0.0f && z == 0.0f) {
      return core::math::matrix::transform::rotate3d_x(angle);
    } else if (x == 0.0f && y == 1.0f && z == 0.0f) {
      return core::math::matrix::transform::rotate3d_y(angle);
    } else if (x == 0.0f && y == 0.0f && z == 1.0f) {
      return core::math::matrix::transform::rotate3d_z(angle);
    }
    throw std::runtime_error("Only axis-aligned rotations supported");
  }

  /**
   * @brief Creates a scaling matrix.
   * @param sx Scaling factor in x direction.
   * @param sy Scaling factor in y direction.
   * @param sz Scaling factor in z direction.
   * @return Scaling matrix.
   */
  static auto create_scale(float sx, float sy, float sz = 1.0f) {
    if (sz == 1.0f) {
      return core::math::matrix::transform::scale2d(sx, sy);
    }
    return core::math::matrix::transform::scale3d(sx, sy, sz);
  }

  /**
   * @brief Creates a look-at matrix for a camera.
   * @param eye Position of the camera.
   * @param target Target position.
   * @param up Up vector.
   * @return Look-at matrix.
   */
  static auto create_look_at(const vector::Vector3& eye,
                             const vector::Vector3& target,
                             const vector::Vector3& up) {
    return core::math::matrix::look_at(eye, target, up);
  }

  /**
   * @brief Creates a perspective projection matrix.
   * @param fov Field of view in radians.
   * @param aspect Aspect ratio.
   * @param near Near clipping plane.
   * @param far Far clipping plane.
   * @return Perspective projection matrix.
   */
  static auto create_perspective(float fov, float aspect, float near,
                                 float far) {
    return core::math::matrix::transform::perspective(fov, aspect, near, far);
  }

  // Utility Functions

  /**
   * @brief Prints a matrix to standard output.
   * @tparam T Type of matrix elements.
   * @param matrix Matrix to print.
   */
  template <typename T>
  static void print(const T& matrix) {
    matrix.print();
  }

  // Sparse Matrix Operations

  /**
   * @brief Creates a sparse matrix.
   * @tparam T Type of matrix elements.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @return Sparse matrix.
   */
  template <typename T>
  static auto create_sparse(size_t rows, size_t cols) {
    return core::math::matrix::SparseMatrix<T>(rows, cols);
  }

  /**
   * @brief Adds two sparse matrices.
   * @tparam T Type of matrix elements.
   * @param a First sparse matrix.
   * @param b Second sparse matrix.
   * @return Result of the addition.
   */
  template <typename T>
  static auto sparse_add(const core::math::matrix::SparseMatrix<T>& a,
                         const core::math::matrix::SparseMatrix<T>& b) {
    return a + b;
  }

  /**
   * @brief Multiplies two sparse matrices.
   * @tparam T Type of matrix elements.
   * @param a First sparse matrix.
   * @param b Second sparse matrix.
   * @return Result of the multiplication.
   */
  template <typename T>
  static auto sparse_multiply(const core::math::matrix::SparseMatrix<T>& a,
                              const core::math::matrix::SparseMatrix<T>& b) {
    return a.multiply(b);
  }

  // Parallel Operations

  /**
   * @brief Multiplies two matrices in parallel.
   * @tparam T Type of matrix elements.
   * @param a First matrix.
   * @param b Second matrix.
   * @return Result of the parallel multiplication.
   */
  template <typename T>
  static auto parallel_multiply(const T& a, const T& b) {
    return core::math::matrix::parallel::parallel_matrix_multiply(a, b);
  }

  // Functional Operations

  /**
   * @brief Applies a function to each element of a matrix.
   * @tparam T Type of matrix elements.
   * @tparam Func Type of the function.
   * @param matrix Matrix to apply the function to.
   * @param func Function to apply.
   * @return Matrix with the function applied to each element.
   */
  template <typename T, typename Func>
  static auto map(const T& matrix, Func func) {
    T result = matrix;
    for (auto& val : result) {
      val = func(val);
    }
    return result;
  }

  /**
   * @brief Reduces a matrix to a single value using a function.
   * @tparam T Type of matrix elements.
   * @tparam Func Type of the function.
   * @param matrix Matrix to reduce.
   * @param func Function to apply.
   * @param initial Initial value.
   * @return Result of the reduction.
   */
  template <typename T, typename Func>
  static auto reduce(const T& matrix, Func func,
                     typename T::value_type initial) {
    typename T::value_type result = initial;
    for (const auto& val : matrix) {
      result = func(result, val);
    }
    return result;
  }
};

// Type aliases for common matrix types
using Matrix2D = core::math::matrix::mat2x2;
using Matrix3D = core::math::matrix::mat3x3;
using Matrix4D = core::math::matrix::mat4x4;
using Vector2D = core::math::matrix::Matrix<float, 2, 1>;
using Vector3D = core::math::vector::Vector3;
using Vector4D = core::math::matrix::Matrix<float, 4, 1>;

}  // namespace core::math::matrix::api
