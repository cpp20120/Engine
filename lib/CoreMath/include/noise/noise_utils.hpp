#pragma once

#include <array>
#include <cmath>
#include <numeric>
#include <random>

#include "../vector.hpp"

namespace core::math::noise {

/**
 * @defgroup Noise Noise Generation
 * @brief Perlin and Simplex noise implementations
 * @{
 */

/// Permutation table size (must be power of 2)
constexpr size_t PERMUTATION_TABLE_SIZE = 256;

/**
 * @brief Base class for gradient noise generators
 */
template <std::size_t N, typename T = float>
class GradientNoise {
 protected:
  /// @name Constructors
  /// @{

  /**
   * @brief Initialize with default permutation table
   */
  GradientNoise() {
    // Fill with 0..255
    std::iota(permutation.begin(), permutation.begin() + PERMUTATION_TABLE_SIZE,
              0);

    // Shuffle the first half
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(permutation.begin(),
                 permutation.begin() + PERMUTATION_TABLE_SIZE, g);

    // Duplicate to avoid modulo
    for (size_t i = 0; i < PERMUTATION_TABLE_SIZE; ++i) {
      permutation[PERMUTATION_TABLE_SIZE + i] = permutation[i];
    }
  }

  /**
   * @brief Initialize with specific seed
   * @param seed Seed value for permutation table
   */
  explicit GradientNoise(uint32_t seed) {
    // Fill with 0..255
    std::iota(permutation.begin(), permutation.begin() + PERMUTATION_TABLE_SIZE,
              0);

    // Shuffle with given seed
    std::mt19937 g(seed);
    std::shuffle(permutation.begin(),
                 permutation.begin() + PERMUTATION_TABLE_SIZE, g);

    // Duplicate to avoid modulo
    for (size_t i = 0; i < PERMUTATION_TABLE_SIZE; ++i) {
      permutation[PERMUTATION_TABLE_SIZE + i] = permutation[i];
    }
  }

  /// @}

  /// @name Utility Functions
  /// @{

  /**
   * @brief Fade function (6t^5 - 15t^4 + 10t^3)
   * @param t Input value
   * @return Faded value
   */
  constexpr T fade(T t) const noexcept {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  /**
   * @brief Linear interpolation
   * @param a First value
   * @param b Second value
   * @param t Interpolation factor [0,1]
   * @return Interpolated value
   */
  constexpr T lerp(T a, T b, T t) const noexcept { return a + t * (b - a); }

  /**
   * @brief Gradient function for Perlin noise
   * @param hash Hash value from permutation table
   * @param point Input point
   * @return Gradient value
   */
  constexpr T grad(int hash, const Vector<N, T>& point) const noexcept {
    if constexpr (N == 2) {
      const T u = hash & 1 ? -point.x() : point.x();
      const T v = hash & 2 ? -point.y() : point.y();
      return u + v;
    } else if constexpr (N == 3) {
      const T u = hash & 1 ? -point.x() : point.x();
      const T v = hash & 4 ? -point.y() : point.y();
      const T w = hash & 8 ? -point.z() : point.z();
      return u + v + w;
    } else {
      // For higher dimensions, use dot product with random gradient
      Vector<N, T> gradient;
      for (size_t i = 0; i < N; ++i) {
        gradient[i] = (hash & (1 << i)) ? -point[i] : point[i];
      }
      return gradient.dot(point);
    }
  }

  /// @}

  std::array<int, PERMUTATION_TABLE_SIZE * 2> permutation;
};

/// @}
}  // namespace core::math::noise