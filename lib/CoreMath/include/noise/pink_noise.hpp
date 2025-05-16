#pragma once
#include <array>
#include <numeric>
#include <random>

namespace core::math::noise {

/**
 * @class PinkNoise
 * @brief A class to generate pink noise.
 *
 * Pink noise, also known as 1/f noise, is a signal or process with a frequency
 * spectrum such that the power spectral density is inversely proportional to
 * the frequency. This class generates pink noise using a filtering approach on
 * white noise.
 *
 * @tparam T The type of the noise values (default is float).
 */
template <typename T = float>
class PinkNoise {
 public:
  /**
   * @brief Constructs a PinkNoise object with a random seed.
   */
  PinkNoise() : gen(std::random_device{}()) { reset(); }

  /**
   * @brief Constructs a PinkNoise object with a specific seed.
   * @param seed The seed for the random number generator.
   */
  explicit PinkNoise(uint32_t seed) : gen(seed) { reset(); }

  /**
   * @brief Generates pink noise for a given input value.
   * @param x The input value (not used directly in this implementation).
   * @return The generated pink noise value.
   */
  T noise(T x) {
    // Generate white noise
    std::uniform_real_distribution<T> dist(-1, 1);
    T white = dist(gen);

    // Apply filter
    b[0] = 0.99886 * b[0] + white * 0.0555179;
    b[1] = 0.99332 * b[1] + white * 0.0750759;
    b[2] = 0.96900 * b[2] + white * 0.1538520;
    b[3] = 0.86650 * b[3] + white * 0.3104856;
    b[4] = 0.55000 * b[4] + white * 0.5329522;
    b[5] = -0.7616 * b[5] - white * 0.0168980;

    T pink = (b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362);
    b[6] = white * 0.115926;

    return pink * 0.11;  // Scale to [-1, 1]
  }

  /**
   * @brief Resets the internal state of the pink noise generator.
   *
   * This function reinitializes the filter coefficients with random values.
   */
  void reset() {
    std::uniform_real_distribution<T> dist(-1, 1);
    for (auto& val : b) {
      val = dist(gen);
    }
  }

 private:
  std::array<T, 7> b; /**< Filter coefficients for generating pink noise. */
  std::mt19937 gen;   /**< Random number generator. */
};

}  // namespace core::math::noise
