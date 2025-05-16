#pragma once
#include "../vector.hpp"
#include "./FastNoiseLite.h"

namespace core::math::noise {

/**
 * @defgroup FastNoise FastNoiseLite Adapter
 * @brief Adapter for FastNoiseLite to match core::math::vector API
 * @{
 */

/**
 * @brief Template class adapting FastNoiseLite to work with Vector types
 * @tparam N Dimension of noise (2 or 3)
 * @tparam T Floating-point type (float or double)
 */
template <std::size_t N, typename T = float>
  requires(N == 2 || N == 3) && core::math::concepts::is_number<T>
class FastNoiseAdapter {
 public:
  using VectorType = Vector<N, T>;

  /// @name Constructors
  /// @{

  /**
   * @brief Construct with default settings
   * @param seed Random seed
   */
  explicit FastNoiseAdapter(int seed = 1337) : m_noise(seed) {
    m_noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
  }

  /**
   * @brief Construct with specific noise type
   * @param seed Random seed
   * @param noiseType Type of noise to generate
   */
  FastNoiseAdapter(int seed, FastNoiseLite::NoiseType noiseType)
      : m_noise(seed) {
    m_noise.SetNoiseType(noiseType);
  }

  /// @}

  /// @name Configuration
  /// @{

  /**
   * @brief Set the noise type
   * @param noiseType Type of noise to generate
   */
  void SetNoiseType(FastNoiseLite::NoiseType noiseType) {
    m_noise.SetNoiseType(noiseType);
  }

  /**
   * @brief Set the fractal type
   * @param fractalType Type of fractal noise
   */
  void SetFractalType(FastNoiseLite::FractalType fractalType) {
    m_noise.SetFractalType(fractalType);
  }

  /**
   * @brief Set number of fractal octaves
   * @param octaves Number of octaves (1-)
   */
  void SetFractalOctaves(int octaves) { m_noise.SetFractalOctaves(octaves); }

  /**
   * @brief Set fractal lacunarity
   * @param lacunarity Frequency multiplier between octaves
   */
  void SetFractalLacunarity(T lacunarity) {
    m_noise.SetFractalLacunarity(static_cast<float>(lacunarity));
  }

  /**
   * @brief Set fractal gain
   * @param gain Amplitude multiplier between octaves
   */
  void SetFractalGain(T gain) {
    m_noise.SetFractalGain(static_cast<float>(gain));
  }

  /**
   * @brief Set frequency
   * @param frequency Noise frequency
   */
  void SetFrequency(T frequency) {
    m_noise.SetFrequency(static_cast<float>(frequency));
  }

  /// @}

  /// @name Noise Generation
  /// @{

  /**
   * @brief Generate noise value at given point
   * @param point Input position
   * @return Noise value in range [-1, 1]
   */
  T noise(const VectorType& point) const {
    if constexpr (N == 2) {
      return static_cast<T>(m_noise.GetNoise(static_cast<float>(point.x()),
                                             static_cast<float>(point.y())));
    } else {
      return static_cast<T>(m_noise.GetNoise(static_cast<float>(point.x()),
                                             static_cast<float>(point.y()),
                                             static_cast<float>(point.z())));
    }
  }

  /**
   * @brief Generate fractal noise by combining multiple octaves
   * @param point Input position
   * @param octaves Number of octaves
   * @param persistence Amplitude multiplier per octave
   * @param lacunarity Frequency multiplier per octave
   * @return Combined noise value
   */
  T fractal(const VectorType& point, int octaves = 5, T persistence = T(0.5),
            T lacunarity = T(2.0)) const {
    // Store current settings
    auto prevOctaves = m_noise.GetFractalOctaves();
    auto prevGain = m_noise.GetFractalGain();
    auto prevLacunarity = m_noise.GetFractalLacunarity();

    // Configure for this fractal call
    m_noise.SetFractalOctaves(octaves);
    m_noise.SetFractalGain(static_cast<float>(persistence));
    m_noise.SetFractalLacunarity(static_cast<float>(lacunarity));

    // Generate noise
    T result = noise(point);

    // Restore settings
    m_noise.SetFractalOctaves(prevOctaves);
    m_noise.SetFractalGain(prevGain);
    m_noise.SetFractalLacunarity(prevLacunarity);

    return result;
  }

  /**
   * @brief Domain warp a position using another noise
   * @param point Point to warp (modified in-place)
   * @param warpAmplitude Strength of the warp effect
   */
  void domainWarp(VectorType& point, T warpAmplitude = T(1.0)) const {
    float x = static_cast<float>(point.x());
    float y = static_cast<float>(point.y());

    if constexpr (N == 2) {
      m_noise.DomainWarp(x, y);
      point.x() = static_cast<T>(x) * warpAmplitude;
      point.y() = static_cast<T>(y) * warpAmplitude;
    } else {
      float z = static_cast<float>(point.z());
      m_noise.DomainWarp(x, y, z);
      point.x() = static_cast<T>(x) * warpAmplitude;
      point.y() = static_cast<T>(y) * warpAmplitude;
      point.z() = static_cast<T>(z) * warpAmplitude;
    }
  }

  /// @}

  /// @name Static Noise Types
  /// @{

  /**
   * @brief Create a Perlin noise generator
   * @param seed Random seed
   * @return Configured noise generator
   */
  static FastNoiseAdapter Perlin(int seed = 1337) {
    FastNoiseAdapter noise(seed);
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    return noise;
  }

  /**
   * @brief Create a Simplex noise generator
   * @param seed Random seed
   * @return Configured noise generator
   */
  static FastNoiseAdapter Simplex(int seed = 1337) {
    FastNoiseAdapter noise(seed);
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    return noise;
  }

  /**
   * @brief Create a Cellular/Voronoi noise generator
   * @param seed Random seed
   * @return Configured noise generator
   */
  static FastNoiseAdapter Cellular(int seed = 1337) {
    FastNoiseAdapter noise(seed);
    noise.SetNoiseType(FastNoiseLite::NoiseType_Cellular);
    return noise;
  }

  /**
   * @brief Create a Value noise generator
   * @param seed Random seed
   * @return Configured noise generator
   */
  static FastNoiseAdapter Value(int seed = 1337) {
    FastNoiseAdapter noise(seed);
    noise.SetNoiseType(FastNoiseLite::NoiseType_Value);
    return noise;
  }

  /// @}

 private:
  mutable FastNoiseLite m_noise;
};

/// Common type aliases
using FastNoise2D = FastNoiseAdapter<2, float>;
using FastNoise3D = FastNoiseAdapter<3, float>;

/// @}
}  // namespace core::math::noise