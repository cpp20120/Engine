#pragma once
#include "./perlin_noise.hpp"

namespace core::math::noise {

/**
 * @class RidgedNoise
 * @brief A class to generate ridged noise in N-dimensional space.
 *
 * Ridged noise is a variation of Perlin noise that produces sharp, ridge-like
 * features. This class extends the PerlinNoise class to generate ridged noise
 * by modifying the output of Perlin noise.
 *
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and noise values (default is float).
 */
template <std::size_t N, typename T = float>
class RidgedNoise : public PerlinNoise<N, T> {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */

  /**
   * @brief Inherit constructors from the base class PerlinNoise.
   */
  using PerlinNoise<N, T>::PerlinNoise;

  /**
   * @brief Generates ridged noise at a given point.
   * @param point The point in N-dimensional space.
   * @return The ridged noise value at the given point.
   */
  T noise(const VectorType& point) const override {
    T val = 1 - std::abs(PerlinNoise<N, T>::noise(point));
    return val * val;
  }
};

}  // namespace core::math::noise
