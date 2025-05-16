#pragma once
#include "./worley_noise.hpp"

namespace core::math::noise {

/**
 * @class VoronoiNoise
 * @brief A class to generate Voronoi noise in N-dimensional space.
 *
 * Voronoi noise is a type of noise that partitions the space into regions based
 * on the distance to feature points. This class extends the WorleyNoise class
 * to provide additional information about the nearest feature point.
 *
 * @tparam N The dimensionality of the noise.
 * @tparam T The type of the coordinates and distances (default is float).
 */
template <std::size_t N, typename T = float>
class VoronoiNoise : public WorleyNoise<N, T, 1> {
 public:
  using VectorType =
      Vector<N, T>; /**< Type alias for an N-dimensional vector. */
  using Base = WorleyNoise<N, T, 1>; /**< Type alias for the base class. */
  using Base::Base; /**< Inherit constructors from the base class. */

  /**
   * @struct Result
   * @brief A structure to hold the result of Voronoi noise generation.
   */
  struct Result {
    T distance;       /**< The distance to the nearest feature point. */
    VectorType point; /**< The position of the nearest feature point. */
    uint32_t id;      /**< A unique identifier for the nearest feature point. */
  };

  /**
   * @brief Generates Voronoi noise at a given point.
   * @param point The point in N-dimensional space.
   * @return A Result structure containing the distance, point, and ID of the
   * nearest feature point.
   */
  Result noise(const VectorType& point) const {
    auto distances = Base::get_distances(point);

    Result result;
    result.distance = distances[0].first;
    result.point = distances[0].second;

    // Generate ID from feature point position
    result.id = 0;
    for (size_t i = 0; i < N; ++i) {
      result.id ^= static_cast<uint32_t>(result.point[i] * 1000) << (i % 16);
    }

    return result;
  }
};

}  // namespace core::math::noise
