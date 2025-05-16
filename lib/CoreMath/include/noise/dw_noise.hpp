#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

/**
 * @class DomainWarpedNoise
 * @brief A class to generate domain-warped noise in N-dimensional space.
 *
 * Domain-warped noise is a type of noise where the input domain is perturbed by
 * another noise function. This class applies a noise function to warp the
 * domain of another noise function, creating more complex and varied noise
 * patterns.
 *
 * @tparam NoiseFn The type of the noise function used for domain warping.
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and noise values (default is float).
 */
template <typename NoiseFn, std::size_t N, typename T = float>
class DomainWarpedNoise {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */

  /**
   * @brief Constructs a DomainWarpedNoise object with a given noise function
   * and amplitude.
   * @param noise The noise function used for domain warping.
   * @param amplitude The amplitude of the domain warping (default is 1.0).
   */
  DomainWarpedNoise(const NoiseFn& noise, T amplitude = 1.0)
      : noise(noise), amplitude(amplitude) {}

  /**
   * @brief Generates domain-warped noise at a given point.
   * @param point The point in N-dimensional space.
   * @return The domain-warped noise value at the given point.
   */
  T noise(const VectorType& point) const {
    VectorType offset;
    for (size_t i = 0; i < N; ++i) {
      offset[i] = amplitude * noise(point);
    }
    return noise(point + offset);
  }

  /**
   * @brief Generates fractal domain-warped noise at a given point.
   * @param point The point in N-dimensional space.
   * @param octaves The number of octaves to use for the fractal noise (default
   * is 4).
   * @param persistence The persistence value for the fractal noise (default is
   * 0.5).
   * @param lacunarity The lacunarity value for the fractal noise (default
   * is 2.0).
   * @return The fractal domain-warped noise value at the given point.
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

 private:
  NoiseFn noise; /**< The noise function used for domain warping. */
  T amplitude;   /**< The amplitude of the domain warping. */
};

}  // namespace core::math::noise
