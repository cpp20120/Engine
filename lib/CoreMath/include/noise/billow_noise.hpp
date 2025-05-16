#pragma once
#include "./perlin_noise.hpp"

namespace core::math::noise {

/**
 * @class BillowNoise
 * @brief A class to generate billow noise in N-dimensional space.
 *
 * Billow noise is a variation of Perlin noise that produces a more "billowy" or
 * cloud-like appearance. This class extends the PerlinNoise class to generate
 * billow noise by modifying the output of Perlin noise.
 *
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and noise values (default is float).
 */
template <std::size_t N, typename T = float>
class BillowNoise : public PerlinNoise<N, T> {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */

  /**
   * @brief Inherit constructors from the base class PerlinNoise.
   */
  using PerlinNoise<N, T>::PerlinNoise;

  /**
   * @brief Generates billow noise at a given point.
   * @param point The point in N-dimensional space.
   * @return The billow noise value at the given point.
   */
  T noise(const VectorType& point) const override {
    return std::abs(PerlinNoise<N, T>::noise(point)) * 2 - 1;
  }
};

}  // namespace core::math::noise
