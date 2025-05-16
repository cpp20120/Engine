#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Include the noise implementations
#include "../noise/billow_noise.hpp"
#include "../noise/curl_noise.hpp"
#include "../noise/dw_noise.hpp"
#include "../noise/fast_noise_adapter.hpp"
#include "../noise/gabor_noise.hpp"
#include "../noise/gradient_noise.hpp"
#include "../noise/perlin_noise.hpp"
#include "../noise/pink_noise.hpp"
#include "../noise/ridged_noise.hpp"
#include "../noise/simplex_noise.hpp"
#include "../noise/value_noise.hpp"
#include "../noise/voronoi_noise.hpp"
#include "../noise/white_noise.hpp"
#include "../noise/worley_noise.hpp"
#include "../vector.hpp"

namespace core::math::noise::api {

/**
 * @class NoiseAPI
 * @brief High-level API for noise generation operations.
 */
class NoiseAPI {
 public:
  // Noise Generation

  /**
   * @brief Creates a Perlin noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Perlin noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_perlin(uint32_t seed = 1337) {
    return core::math::noise::PerlinNoise<N, T>(seed);
  }

  /**
   * @brief Creates a Simplex noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Simplex noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_simplex(uint32_t seed = 1337) {
    return core::math::noise::SimplexNoise<N, T>(seed);
  }

  /**
   * @brief Creates a Value noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Value noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_value(uint32_t seed = 1337) {
    return core::math::noise::ValueNoise<N, T>(seed);
  }

  /**
   * @brief Creates a Worley noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @tparam K Number of feature points to consider.
   * @param seed Seed for the random number generator.
   * @return Worley noise generator.
   */
  template <std::size_t N, typename T = float, size_t K = 1>
  static auto create_worley(uint32_t seed = 1337) {
    return core::math::noise::WorleyNoise<N, T, K>(seed);
  }

  /**
   * @brief Creates a Billow noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Billow noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_billow(uint32_t seed = 1337) {
    return core::math::noise::BillowNoise<N, T>(seed);
  }

  /**
   * @brief Creates a Ridged noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Ridged noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_ridged(uint32_t seed = 1337) {
    return core::math::noise::RidgedNoise<N, T>(seed);
  }

  /**
   * @brief Creates a Curl noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param scale Scale of the noise.
   * @return Curl noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_curl(T scale = 1.0) {
    static_assert(N == 2 || N == 3,
                  "Curl noise only implemented for 2D and 3D");
    return core::math::noise::CurlNoise<N, T>(scale);
  }

  /**
   * @brief Creates a Voronoi noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Voronoi noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_voronoi(uint32_t seed = 1337) {
    return core::math::noise::VoronoiNoise<N, T>(seed);
  }

  /**
   * @brief Creates a DomainWarped noise generator.
   * @tparam NoiseFn Type of the noise function.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param noise Noise function to warp.
   * @param amplitude Amplitude of the warp.
   * @return DomainWarped noise generator.
   */
  template <typename NoiseFn, std::size_t N, typename T = float>
  static auto create_domain_warped(const NoiseFn& noise, T amplitude = 1.0) {
    return core::math::noise::DomainWarpedNoise<NoiseFn, N, T>(noise,
                                                               amplitude);
  }

  /**
   * @brief Creates a Gabor noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param kernel_size Size of the kernel.
   * @param frequency Frequency of the noise.
   * @return Gabor noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_gabor(T kernel_size = 0.1, T frequency = 10.0) {
    return core::math::noise::GaborNoise<N, T>(kernel_size, frequency);
  }

  /**
   * @brief Creates a FastNoise adapter.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return FastNoise adapter.
   */
  template <std::size_t N, typename T = float>
  static auto create_fast_noise(int seed = 1337) {
    static_assert(N == 2 || N == 3, "FastNoise only implemented for 2D and 3D");
    return core::math::noise::FastNoiseAdapter<N, T>(seed);
  }

  /**
   * @brief Creates a Pink noise generator.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return Pink noise generator.
   */
  template <typename T = float>
  static auto create_pink(uint32_t seed = std::random_device{}()) {
    return core::math::noise::PinkNoise<T>(seed);
  }

  /**
   * @brief Creates a White noise generator.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param seed Seed for the random number generator.
   * @return White noise generator.
   */
  template <std::size_t N, typename T = float>
  static auto create_white(uint32_t seed = std::random_device{}()) {
    return core::math::noise::WhiteNoise<N, T>(seed);
  }

  // Noise Operations

  /**
   * @brief Generates noise at a given point.
   * @tparam NoiseFn Type of the noise function.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param noise Noise generator.
   * @param point Point at which to generate noise.
   * @return Noise value at the given point.
   */
  template <typename NoiseFn, std::size_t N, typename T>
  static auto noise(const NoiseFn& noise, const Vector<N, T>& point) {
    return noise.noise(point);
  }

  /**
   * @brief Generates fractal noise at a given point.
   * @tparam NoiseFn Type of the noise function.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param noise Noise generator.
   * @param point Point at which to generate noise.
   * @param octaves Number of octaves.
   * @param persistence Persistence of the noise.
   * @param lacunarity Lacunarity of the noise.
   * @return Fractal noise value at the given point.
   */
  template <typename NoiseFn, std::size_t N, typename T>
  static auto fractal(const NoiseFn& noise, const Vector<N, T>& point,
                      size_t octaves = 4, T persistence = T(0.5),
                      T lacunarity = T(2.0)) {
    return noise.fractal(point, octaves, persistence, lacunarity);
  }

  /**
   * @brief Domain warps a point using another noise.
   * @tparam NoiseFn Type of the noise function.
   * @tparam N Dimension of the noise.
   * @tparam T Type of the noise values.
   * @param noise Noise generator.
   * @param point Point to warp.
   * @param warp_amplitude Strength of the warp effect.
   */
  template <typename NoiseFn, std::size_t N, typename T>
  static void domain_warp(NoiseFn& noise, Vector<N, T>& point,
                          T warp_amplitude = T(1.0)) {
    noise.domainWarp(point, warp_amplitude);
  }
};
}  // namespace core::math::noise::api
