#pragma once
#include "./worley_noise.hpp"

namespace core::math::noise {

template <std::size_t N, typename T = float>
class VoronoiNoise : public WorleyNoise<N, T, 1> {
 public:
  using VectorType = Vector<N, T>;
  using Base = WorleyNoise<N, T, 1>;
  using Base::Base;

  struct Result {
    T distance;
    VectorType point;
    uint32_t id;
  };

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