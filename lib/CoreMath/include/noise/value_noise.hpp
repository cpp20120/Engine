#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

/**
 * @class ValueNoise
 * @brief A class to generate value noise in N-dimensional space.
 *
 * Value noise is a type of gradient noise that generates interpolated random
 * values. This class extends the GradientNoise class to provide value noise
 * generation.
 *
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and noise values (default is float).
 */
template <std::size_t N, typename T = float>
class ValueNoise : public GradientNoise<N, T> {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */

  /// @name Constructors
  /// @{
  /**
   * @brief Constructs a ValueNoise object with a default seed.
   */
  ValueNoise() : GradientNoise<N, T>() {}

  /**
   * @brief Constructs a ValueNoise object with a specific seed.
   * @param seed The seed for the random number generator.
   */
  explicit ValueNoise(uint32_t seed) : GradientNoise<N, T>(seed) {}
  /// @}

  /**
   * @brief Generates value noise at a given point.
   * @param point The point in N-dimensional space.
   * @return The noise value at the given point, scaled to the range [-1, 1].
   */
  T noise(const VectorType& point) const override {
    std::array<int, N> cube;
    VectorType frac;

    for (size_t i = 0; i < N; ++i) {
      cube[i] = static_cast<int>(std::floor(point[i]));
      frac[i] = point[i] - cube[i];
      cube[i] &= (PERMUTATION_TABLE_SIZE - 1);
    }

    VectorType u;
    for (size_t i = 0; i < N; ++i) {
      u[i] = this->fade(frac[i]);
    }

    T result = 0;

    for (size_t corner = 0; corner < (1 << N); ++corner) {
      std::array<int, N> vertex;
      T weight = 1;
      int hash = 0;

      for (size_t i = 0; i < N; ++i) {
        const int corner_bit = (corner >> i) & 1;
        vertex[i] = cube[i] + corner_bit;
        hash ^= this->permutation[vertex[i]] << i;
        weight *= (corner_bit ? u[i] : (1 - u[i]));
      }

      // Get random value from hash
      T val = static_cast<T>(hash % 10000) / 10000.0f;
      result += weight * val;
    }

    return result * 2 - 1;  // Scale to [-1, 1]
  }

  /**
   * @brief Generates fractal value noise at a given point.
   * @param point The point in N-dimensional space.
   * @param octaves The number of octaves to use for the fractal noise (default
   * is 4).
   * @param persistence The persistence value for the fractal noise (default is
   * 0.5).
   * @param lacunarity The lacunarity value for the fractal noise (default
   * is 2.0).
   * @return The fractal noise value at the given point.
   */
  T fractal(const VectorType& point, size_t octaves = 4, T persistence = T(0.5),
            T lacunarity = T(2.0)) const {
    T result = 0;
    T amplitude = 1;
    T frequency = 1;
    T max_value = 0;

    VectorType current_point = point;

    for (size_t i = 0; i < octaves; ++i) {
      result += noise(current_point) * amplitude;
      max_value += amplitude;

      amplitude *= persistence;
      frequency *= lacunarity;
      current_point = point * frequency;
    }

    return result / max_value;
  }
};

}  // namespace core::math::noise
