#pragma once

#include <array>
#include <initializer_list>
#include <vector>

#include "./concepts.hpp"
#include "./parallel/parallel_executor.hpp"
#include "./vector.hpp"

namespace core::math::vector::api {

/**
 * @class VectorAPI
 * @brief High-level API for vector operations.
 */
class VectorAPI {
 public:
  // Vector Creation

  /**
   * @brief Creates a vector with the specified components.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param components Components of the vector.
   * @return Vector with the specified components.
   */
  template <std::size_t N, typename T>
  static auto create(const std::array<T, N>& components) {
    return Vector<N, T>(components);
  }

  /**
   * @brief Creates a vector from an initializer list.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param init Initializer list of components.
   * @return Vector with the specified components.
   */
  template <std::size_t N, typename T>
  static auto create(std::initializer_list<T> init) {
    return Vector<N, T>(init);
  }

  /**
   * @brief Creates a vector with variadic arguments.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @tparam Args Types of the variadic arguments.
   * @param args Components of the vector.
   * @return Vector with the specified components.
   */
  template <std::size_t N, typename T, typename... Args>
  static auto create(Args... args) {
    return Vector<N, T>(args...);
  }

  // Vector Operations

  /**
   * @brief Adds two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Result of the addition.
   */
  template <std::size_t N, typename T>
  static auto add(const Vector<N, T>& a, const Vector<N, T>& b) {
    return a + b;
  }

  /**
   * @brief Subtracts two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Result of the subtraction.
   */
  template <std::size_t N, typename T>
  static auto subtract(const Vector<N, T>& a, const Vector<N, T>& b) {
    return a - b;
  }

  /**
   * @brief Multiplies a vector by a scalar.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param vector Vector to multiply.
   * @param scalar Scalar value.
   * @return Result of the multiplication.
   */
  template <std::size_t N, typename T>
  static auto multiply(const Vector<N, T>& vector, T scalar) {
    return vector * scalar;
  }

  /**
   * @brief Divides a vector by a scalar.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param vector Vector to divide.
   * @param scalar Scalar value.
   * @return Result of the division.
   * @throws std::domain_error if scalar is zero.
   */
  template <std::size_t N, typename T>
  static auto divide(const Vector<N, T>& vector, T scalar) {
    return vector / scalar;
  }

  /**
   * @brief Computes the dot product of two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Dot product of the vectors.
   */
  template <std::size_t N, typename T>
  static auto dot(const Vector<N, T>& a, const Vector<N, T>& b) {
    return a.dot(b);
  }

  /**
   * @brief Computes the cross product of two 3D vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Cross product of the vectors.
   */
  template <typename T>
  static auto cross(const Vector<3, T>& a, const Vector<3, T>& b) {
    return a.cross(b);
  }

  /**
   * @brief Computes the magnitude of a vector.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param vector Vector to compute magnitude.
   * @return Magnitude of the vector.
   */
  template <std::size_t N, typename T>
  static auto magnitude(const Vector<N, T>& vector) {
    return vector.magnitude();
  }

  /**
   * @brief Normalizes a vector.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param vector Vector to normalize.
   * @return Normalized vector.
   * @throws std::domain_error if magnitude is zero.
   */
  template <std::size_t N, typename T>
  static auto normalize(const Vector<N, T>& vector) {
    return vector.normalize();
  }

  /**
   * @brief Computes the Hadamard product of two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Hadamard product of the vectors.
   */
  template <std::size_t N, typename T>
  static auto hadamard(const Vector<N, T>& a, const Vector<N, T>& b) {
    return a.hadamard(b);
  }

  /**
   * @brief Computes the squared magnitude of a vector.
   * @tparam N Dimension of the vector.
   * @tparam T Type of vector components.
   * @param vector Vector to compute squared magnitude.
   * @return Squared magnitude of the vector.
   */
  template <std::size_t N, typename T>
  static auto magnitude_squared(const Vector<N, T>& vector) {
    return vector.magnitude_squared();
  }

  /**
   * @brief Checks if two vectors are approximately equal.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @param epsilon The allowable difference per component (default: 1e-6).
   * @return True if vectors are approximately equal, false otherwise.
   */
  template <std::size_t N, typename T>
  static auto approx_equal(const Vector<N, T>& a, const Vector<N, T>& b,
                           T epsilon = 1e-6) {
    return a.approx_equal(b, epsilon);
  }

  /**
   * @brief Computes the Euclidean distance between two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Euclidean distance between the vectors.
   */
  template <std::size_t N, typename T>
  static auto distance(const Vector<N, T>& a, const Vector<N, T>& b) {
    return a.distance(b);
  }

  /**
   * @brief Performs linear interpolation between two vectors.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a The starting vector (t = 0).
   * @param b The ending vector (t = 1).
   * @param t Interpolation parameter (typically in [0,1]).
   * @return The interpolated vector.
   */
  template <std::size_t N, typename T>
  static auto lerp(const Vector<N, T>& a, const Vector<N, T>& b, T t) {
    return Vector<N, T>::lerp(a, b, t);
  }

  /**
   * @brief Creates a sliced subvector.
   * @tparam N Dimension of the vector.
   * @tparam Start Starting index of the slice.
   * @tparam End Ending index of the slice (exclusive).
   * @tparam T Type of vector components.
   * @param vector Vector to slice.
   * @return New vector containing the sliced components.
   */
  template <std::size_t N, std::size_t Start, std::size_t End, typename T>
  static auto slice(const Vector<N, T>& vector) {
    return vector.template slice<Start, End>();
  }

  // Parallel Vector Operations

  /**
   * @brief Adds two vectors in parallel.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param result Result vector.
   * @param a First vector.
   * @param b Second vector.
   */
  template <std::size_t N, typename T>
  static void parallel_add(Vector<N, T>& result, const Vector<N, T>& a,
                           const Vector<N, T>& b) {
    parallel::parallel_vector_add(result, a, b);
  }

  /**
   * @brief Computes the dot product of two vectors in parallel.
   * @tparam N Dimension of the vectors.
   * @tparam T Type of vector components.
   * @param a First vector.
   * @param b Second vector.
   * @return Dot product of the vectors.
   */
  template <std::size_t N, typename T>
  static auto parallel_dot(const Vector<N, T>& a, const Vector<N, T>& b) {
    return parallel::parallel_vector_dot(a, b);
  }
};

}  // namespace core::math::vector::api
