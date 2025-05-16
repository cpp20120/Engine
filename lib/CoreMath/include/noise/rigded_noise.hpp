#pragma once
#include "./perlin_noise.hpp"

namespace core::math::noise {

template <std::size_t N, typename T = float>
class RidgedNoise : public PerlinNoise<N, T> {
 public:
  using VectorType = Vector<N, T>;

  using PerlinNoise<N, T>::PerlinNoise;

  T noise(const VectorType& point) const override {
    T val = 1 - std::abs(PerlinNoise<N, T>::noise(point));
    return val * val;
  }
};

}  // namespace core::math::noise