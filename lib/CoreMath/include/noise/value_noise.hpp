#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

/**
 * @brief Value noise generator (interpolated random values)
 * @tparam N Dimension of noise
 * @tparam T Floating-point type
 */
template <std::size_t N, typename T = float>
class ValueNoise : public GradientNoise<N, T> {
 public:
  using VectorType = Vector<N, T>;

  /// @name Constructors
  /// @{
  ValueNoise() : GradientNoise<N, T>() {}
  explicit ValueNoise(uint32_t seed) : GradientNoise<N, T>(seed) {}
  /// @}

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