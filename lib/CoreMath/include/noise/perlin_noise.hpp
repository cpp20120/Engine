#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

/**
 * @brief Perlin noise generator
 * @tparam N Dimension of noise (typically 2, 3, or 4)
 * @tparam T Floating-point type (float or double)
 */
template <std::size_t N, typename T = float>
class PerlinNoise : public GradientNoise<N, T> {
 public:
  using VectorType = Vector<N, T>;

  /// @name Constructors
  /// @{

  /**
   * @brief Construct with default random permutation table
   */
  PerlinNoise() : GradientNoise<N, T>() {}

  /**
   * @brief Construct with specific seed
   * @param seed Seed value for permutation table
   */
  explicit PerlinNoise(uint32_t seed) : GradientNoise<N, T>(seed) {}

  /// @}

  /// @name Noise Generation
  /// @{

  /**
   * @brief Generate Perlin noise at given point
   * @param point Input point
   * @return Noise value in range [-1, 1]
   */
  T noise(const VectorType& point) const {
    // Find unit hypercube that contains point
    std::array<int, N> cube;
    for (size_t i = 0; i < N; ++i) {
      cube[i] =
          static_cast<int>(std::floor(point[i])) & (PERMUTATION_TABLE_SIZE - 1);
    }

    // Get fractional part of point
    VectorType frac;
    for (size_t i = 0; i < N; ++i) {
      frac[i] = point[i] - std::floor(point[i]);
    }

    // Compute fade curves
    VectorType u;
    for (size_t i = 0; i < N; ++i) {
      u[i] = this->fade(frac[i]);
    }

    T result = 0;

    // Process all corners of hypercube
    for (size_t corner = 0; corner < (1 << N); ++corner) {
      VectorType weight;
      int hash = 0;

      for (size_t i = 0; i < N; ++i) {
        const int corner_bit = (corner >> i) & 1;
        const int cube_idx = cube[i] + corner_bit;
        hash ^= this->permutation[cube_idx] << i;

        weight[i] = corner_bit ? u[i] : (1 - u[i]);
      }

      // Compute contribution
      VectorType gradient_point = frac;
      for (size_t i = 0; i < N; ++i) {
        const int corner_bit = (corner >> i) & 1;
        gradient_point[i] -= corner_bit;
      }

      const T grad_val = this->grad(hash, gradient_point);
      const T contrib = std::accumulate(weight.begin(), weight.end(), T(1),
                                        std::multiplies<T>()) *
                        grad_val;
      result += contrib;
    }

    // Normalize to [-1, 1]
    if constexpr (N == 2) {
      return result * T(1.41421356237);  // sqrt(2)
    } else if constexpr (N == 3) {
      return result * T(1.15470053838);  // sqrt(3/2)
    } else {
      return result * T(0.5) * std::sqrt(static_cast<T>(N));
    }
  }

  /**
   * @brief Generate fractal noise by combining multiple octaves
   * @param point Input point
   * @param octaves Number of octaves
   * @param persistence Amplitude multiplier per octave
   * @param lacunarity Frequency multiplier per octave
   * @return Combined noise value
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

  /// @}
};

/// Common type aliases
using PerlinNoise2D = PerlinNoise<2, float>;
using PerlinNoise3D = PerlinNoise<3, float>;
using PerlinNoise4D = PerlinNoise<4, float>;

}  // namespace core::math::noise