#pragma once
#include <array>
#include <initializer_list>

#include "./concepts.hpp"
#include "./parallel/parallel_executor.hpp"

namespace core::math::vector {
/**
 * @defgroup Vector Math Vector
 * @brief N-dimensional vector implementation with common mathematical
 * operations
 * @{
 */

/**
 * @brief Template class representing an N-dimensional mathematical vector
 * @tparam N Dimension of the vector
 * @tparam T Arithmetic type of vector components
 *
 * This class provides a generic implementation of mathematical vectors with
 * common operations like addition, subtraction, dot product, normalization,
 * etc. It includes swizzle operations for the first 4 components (x,y,z,w) when
 * available.
 */
template <std::size_t N, typename T>
  requires core::math::concepts::is_number<T>
class Vector {
 public:
  /// @name Constructors
  /// @{

  /**
   * @brief Default constructor (initializes all components to zero)
   */
  constexpr Vector() : data{} {}

  /**
   * @brief Construct from std::array
   * @param init Array containing initial values for all components
   */
  constexpr Vector(const std::array<T, N>& init) : data(init) {}

  /**
   * @brief Construct from initializer list
   * @param init List of initial values (truncated to N elements if too long)
   */
  constexpr Vector(std::initializer_list<T> init) {
    std::size_t i = 0;
    for (const auto& val : init) {
      if (i < N) {
        data[i++] = val;
      }
    }
  }

  /**
   * @brief Variadic constructor for direct component initialization
   * @tparam Args Pack of N convertible-to-T types
   * @param args Values for each component
   */
  template <typename... Args>
    requires(sizeof...(Args) == N && (std::is_convertible_v<Args, T> && ...))
  constexpr Vector(Args... args) : data{static_cast<T>(args)...} {}

  constexpr auto begin() noexcept { return data.begin(); }
  constexpr auto end() noexcept { return data.end(); }

  /// @}

  /// @name Element Access
  /// @{

  /**
   * @brief Subscript operator (mutable)
   * @param index Component index (0-based)
   * @return Reference to the requested component
   */
  constexpr T& operator[](std::size_t index) { return data[index]; }

  /**
   * @brief Subscript operator (const)
   * @param index Component index (0-based)
   * @return Const reference to the requested component
   */
  constexpr const T& operator[](std::size_t index) const { return data[index]; }

  /// @}

  /// @name Swizzle Operations
  /// @{

  /**
   * @brief Access first component (x) - mutable
   * @return Reference to x component
   * @pre N > 0
   */
  constexpr T& x()
    requires(N > 0)
  {
    return data[0];
  }

  /**
   * @brief Access first component (x) - const
   * @return Const reference to x component
   * @pre N > 0
   */
  constexpr const T& x() const
    requires(N > 0)
  {
    return data[0];
  }

  /**
   * @brief Access second component (y) - mutable
   * @return Reference to y component
   * @pre N > 1
   */
  constexpr T& y()
    requires(N > 1)
  {
    return data[1];
  }

  /**
   * @brief Access second component (y) - const
   * @return Const reference to y component
   * @pre N > 1
   */
  constexpr const T& y() const
    requires(N > 1)
  {
    return data[1];
  }

  /**
   * @brief Access third component (z) - mutable
   * @return Reference to z component
   * @pre N > 2
   */
  constexpr T& z()
    requires(N > 2)
  {
    return data[2];
  }

  /**
   * @brief Access third component (z) - const
   * @return Const reference to z component
   * @pre N > 2
   */
  constexpr const T& z() const
    requires(N > 2)
  {
    return data[2];
  }

  /**
   * @brief Access fourth component (w) - mutable
   * @return Reference to w component
   * @pre N > 3
   */
  constexpr T& w()
    requires(N > 3)
  {
    return data[3];
  }

  /**
   * @brief Access fourth component (w) - const
   * @return Const reference to w component
   * @pre N > 3
   */
  constexpr const T& w() const
    requires(N > 3)
  {
    return data[3];
  }

  /// @}

  /// @name Arithmetic Operations
  /// @{

  /**
   * @brief Vector addition
   * @param other Vector to add
   * @return New vector containing component-wise sum
   */
  constexpr Vector operator+(const Vector& other) const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = data[i] + other[i];
    }
    return result;
  }

  /**
   * @brief Vector subtraction
   * @param other Vector to subtract
   * @return New vector containing component-wise difference
   */
  constexpr Vector operator-(const Vector& other) const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = data[i] - other[i];
    }
    return result;
  }

  /**
   * @brief Scalar multiplication
   * @param scalar Value to multiply by
   * @return New vector with each component multiplied by scalar
   */
  constexpr Vector operator*(T scalar) const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = data[i] * scalar;
    }
    return result;
  }

  /**
   * @brief Scalar division
   * @param scalar Value to divide by
   * @return New vector with each component divided by scalar
   * @throws std::domain_error if scalar is zero
   */
  constexpr Vector operator/(T scalar) const {
    if (scalar == T{}) {
      throw std::domain_error("Division by zero");
    }
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = data[i] / scalar;
    }
    return result;
  }

  /**
   * @brief Compound vector addition
   * @param other Vector to add
   * @return Reference to this vector after addition
   */
  constexpr Vector& operator+=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      data[i] += other[i];
    }
    return *this;
  }

  /**
   * @brief Compound vector subtraction
   * @param other Vector to subtract
   * @return Reference to this vector after subtraction
   */
  constexpr Vector& operator-=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      data[i] -= other[i];
    }
    return *this;
  }

  /**
   * @brief Compound scalar multiplication
   * @param scalar Value to multiply by
   * @return Reference to this vector after multiplication
   */
  constexpr Vector& operator*=(T scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      data[i] *= scalar;
    }
    return *this;
  }

  /**
   * @brief Compound scalar division
   * @param scalar Value to divide by
   * @return Reference to this vector after division
   * @throws std::domain_error if scalar is zero
   */
  constexpr Vector& operator/=(T scalar) {
    if (scalar == T{}) {
      throw std::domain_error("Division by zero");
    }
    for (std::size_t i = 0; i < N; ++i) {
      data[i] /= scalar;
    }
    return *this;
  }

  /// @}

  /// @name Vector Operations
  /// @{

  /**
   * @brief Dot product (inner product)
   * @param other Vector to compute dot product with
   * @return Dot product value
   */
  constexpr T dot(const Vector& other) const {
    T result = T{};
    for (std::size_t i = 0; i < N; ++i) {
      result += data[i] * other[i];
    }
    return result;
  }

  /**
   * @brief Compute vector magnitude (length)
   * @return Euclidean norm of the vector
   */
  constexpr T magnitude() const { return std::sqrt(this->dot(*this)); }

  /**
   * @brief Normalize the vector (create unit vector)
   * @return New vector in same direction with length 1
   * @throws std::domain_error if magnitude is zero
   */
  constexpr Vector normalize() const { return *this / this->magnitude(); }

  /// @}

  /// @name Static Operations
  /// @{

  /**
   * @brief Component-wise maximum of two vectors
   * @param a First vector
   * @param b Second vector
   * @return New vector with each component being max of a and b's components
   */
  static constexpr Vector max(const Vector& a, const Vector& b) {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = std::max(a[i], b[i]);
    }
    return result;
  }

  /**
   * @brief Component-wise minimum of two vectors
   * @param a First vector
   * @param b Second vector
   * @return New vector with each component being min of a and b's components
   */
  static constexpr Vector min(const Vector& a, const Vector& b) {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = std::min(a[i], b[i]);
    }
    return result;
  }
  /**
   * @brief Cross Product
   * @param other const ref to vector
   * @return Cross product of vector
   */
  constexpr Vector cross(const Vector& other) const
    requires(N == 3)
  {
    return Vector{y() * other.z() - z() * other.y(),
                  z() * other.x() - x() * other.z(),
                  x() * other.y() - y() * other.x()};
  }
  /**
   * @brief Hadamard product
   * @param other const ref to vector
   * @return result of Hadamard product
   */
  constexpr Vector hadamard(const Vector& other) const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = data[i] * other[i];
    }
    return result;
  }
  /**
   * @brief Computes the squared magnitude (length) of the vector
   * @return The dot product of the vector with itself (avoiding sqrt
   * computation)
   *
   * This is faster than magnitude() and useful for comparisons where
   * the actual length isn't needed, only relative comparisons.
   */
  constexpr T magnitude_squared() const { return this->dot(*this); }

  /**
   * @brief Checks if two vectors are approximately equal within a tolerance
   * @param other The vector to compare against
   * @param epsilon The allowable difference per component (default: 1e-6)
   * @return true if all components differ by no more than epsilon, false
   * otherwise
   *
   * Useful for floating-point comparisons where exact equality is unlikely.
   * For integer types, consider using operator== instead.
   */
  bool approx_equal(const Vector& other, T epsilon = 1e-6) const {
    for (std::size_t i = 0; i < N; ++i) {
      if (std::abs(data[i] - other[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Computes the Euclidean distance between two vectors
   * @param other The target vector
   * @return The distance between this vector and the other vector
   *
   * Equivalent to (this - other).magnitude().
   */
  constexpr T distance(const Vector& other) const {
    return (*this - other).magnitude();
  }

  /**
   * @brief Performs linear interpolation between two vectors
   * @param a The starting vector (t = 0)
   * @param b The ending vector (t = 1)
   * @param t Interpolation parameter (typically in [0,1])
   * @return The interpolated vector
   *
   * When t=0, returns a. When t=1, returns b. Values outside [0,1] extrapolate.
   */
  static constexpr Vector lerp(const Vector& a, const Vector& b, T t) {
    return a + (b - a) * t;
  }

  /**
   * @brief Creates a sliced subvector
   * @tparam Start Starting index of the slice
   * @tparam End Ending index of the slice (exclusive)
   * @return New vector containing the sliced components
   */
  template <size_t Start, size_t End>
  constexpr Vector<End - Start, T> slice() const {
    static_assert(Start < End, "Start must be less than End");
    static_assert(End <= N, "End cannot exceed vector dimension");

    Vector<End - Start, T> result;
    for (size_t i = Start; i < End; ++i) {
      result[i - Start] = data[i];
    }
    return result;
  }

  /// @}

 private:
  std::array<T, N> data;  ///< Internal storage for vector components
};

namespace parallel {
template <std::size_t N, typename T>
void parallel_vector_add(Vector<N, T>& result, const Vector<N, T>& a,
                         const Vector<N, T>& b) {
  core::math::parallel::parallel_for(
      size_t(0), N, [&](size_t i) { result[i] = a[i] + b[i]; });
}

template <std::size_t N, typename T>
T parallel_vector_dot(const Vector<N, T>& a, const Vector<N, T>& b) {
  T result = T{};
  core::math::parallel::parallel_reduce(
      size_t(0), N, [&](size_t i) { return a[i] * b[i]; }, std::plus<T>(),
      result);
  return result;
}
}  // namespace parallel

/// @name Common Vector Type Aliases
/// @{

using Vector2 = Vector<2, float>;  ///< 2D float vector
using Vector3 = Vector<3, float>;  ///< 3D float vector
using Vector4 = Vector<4, float>;  ///< 4D float vector

/// @}
/// @}
}  // namespace core::math::vector