#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

/**
 * @brief Simplex noise generator
 * @tparam N Dimension of noise (typically 2, 3, or 4)
 * @tparam T Floating-point type (float or double)
 */
template <std::size_t N, typename T = float>
class SimplexNoise : public GradientNoise<N, T> {
 public:
  using VectorType = Vector<N, T>;

  /// @name Constructors
  /// @{

  /**
   * @brief Construct with default random permutation table
   */
  SimplexNoise() : GradientNoise<N, T>() { init_gradients(); }

  /**
   * @brief Construct with specific seed
   * @param seed Seed value for permutation table
   */
  explicit SimplexNoise(uint32_t seed) : GradientNoise<N, T>(seed) {
    init_gradients();
  }

  /// @}

  /// @name Noise Generation
  /// @{

  /**
   * @brief Generate Simplex noise at given point
   * @param point Input point
   * @return Noise value in range [-1, 1]
   */
  T noise(VectorType point) const {
    // Skewing factor for N dimensions
    const T F = (std::sqrt(static_cast<T>(N + 1)) - 1) / static_cast<T>(N);

    // Unskewing factor
    const T G = (static_cast<T>(1) - 1 / std::sqrt(static_cast<T>(N + 1))) /
                static_cast<T>(N);

    // Skew the input space to determine which simplex cell we're in
    T skew_sum = std::accumulate(point.begin(), point.end(), T(0));
    const T s = skew_sum * F;

    VectorType skewed;
    for (size_t i = 0; i < N; ++i) {
      skewed[i] = point[i] + s;
    }

    // Determine simplex cell origin
    std::array<int, N> origin;
    for (size_t i = 0; i < N; ++i) {
      origin[i] = static_cast<int>(std::floor(skewed[i]));
    }

    // Unskew the cell origin back to original space
    T unskew_sum = std::accumulate(origin.begin(), origin.end(), T(0));
    const T t = unskew_sum * G;

    VectorType unskewed_origin;
    for (size_t i = 0; i < N; ++i) {
      unskewed_origin[i] = static_cast<T>(origin[i]) - t;
    }

    // The offset from the cell origin to the input point
    VectorType offset = point - unskewed_origin;

    // Determine which simplex we're in (find the largest component)
    std::array<size_t, N> simplex_order;
    for (size_t i = 0; i < N; ++i) {
      simplex_order[i] = i;
    }

    // Sort dimensions by decreasing offset
    std::sort(simplex_order.begin(), simplex_order.end(),
              [&offset](size_t a, size_t b) { return offset[a] > offset[b]; });

    // Traverse the simplex in order of decreasing offset
    T result = 0;
    for (size_t i = 0; i <= N; ++i) {
      // Current vertex of simplex
      std::array<int, N> vertex = origin;
      for (size_t j = 0; j < i; ++j) {
        vertex[simplex_order[j]] += 1;
      }

      // Offset from vertex to input point
      VectorType vertex_offset = point;
      for (size_t k = 0; k < N; ++k) {
        vertex_offset[k] -= static_cast<T>(vertex[k]);
      }

      // Calculate contribution
      T t = T(0.5) - vertex_offset.dot(vertex_offset);
      if (t > 0) {
        t *= t;
        t *= t;

        // Get gradient
        int hash = 0;
        for (size_t k = 0; k < N; ++k) {
          hash ^= this->permutation[vertex[k] & (PERMUTATION_TABLE_SIZE - 1)]
                  << k;
        }

        const T grad_val = this->grad(hash, vertex_offset);
        result += t * grad_val;
      }
    }

    // Scale to [-1, 1]
    const T scale =
        static_cast<T>(N == 2   ? 70.0
                       : N == 3 ? 32.0
                       : N == 4 ? 27.0
                                : 1.0 / std::sqrt(static_cast<T>(N)));
    return result * scale;
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

 private:
  void init_gradients() {
    // For 2D and 3D we can use predefined gradients for better quality
    if constexpr (N == 2 || N == 3) {
      for (size_t i = 0; i < PERMUTATION_TABLE_SIZE; ++i) {
        this->permutation[i] &= (1 << N) - 1;
      }
    }
  }
};

/// Common type aliases
using SimplexNoise2D = SimplexNoise<2, float>;
using SimplexNoise3D = SimplexNoise<3, float>;
using SimplexNoise4D = SimplexNoise<4, float>;

}  // namespace core::math::noise