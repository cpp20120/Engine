#pragma once
#include <random>

namespace core::math::noise {

/**
 * @class WhiteNoise
 * @brief A class to generate white noise in N-dimensional space.
 *
 * White noise is a random signal having equal intensity at different
 * frequencies, giving it a constant power spectral density. This implementation
 * generates white noise by hashing the coordinates of a point to produce a seed
 * for a random number generator.
 *
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and noise values (default is float).
 */
template <std::size_t N, typename T = float>
class WhiteNoise {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */

  /**
   * @brief Constructs a WhiteNoise object with a random seed.
   *
   * Initializes the random number generator with a seed obtained from a random
   * device.
   */
  WhiteNoise() : gen(std::random_device{}()) {}

  /**
   * @brief Constructs a WhiteNoise object with a specific seed.
   * @param seed The seed for the random number generator.
   */
  explicit WhiteNoise(uint32_t seed) : gen(seed) {}

  /**
   * @brief Generates white noise at a given point.
   * @param point The point in N-dimensional space.
   * @return A random noise value in the range [-1, 1].
   */
  T noise(const VectorType& point) const {
    // Hash the point coordinates to get a seed
    uint32_t seed = 0;
    for (size_t i = 0; i < N; ++i) {
      seed ^= static_cast<uint32_t>(point[i] * 1000) << (i % 16);
    }

    // Use the seed to generate a random value
    std::mt19937 local_gen(seed);
    std::uniform_real_distribution<T> dist(-1, 1);
    return dist(local_gen);
  }

 private:
  mutable std::mt19937 gen; /**< Random number generator. */
};

}  // namespace core::math::noise
