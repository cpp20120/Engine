#pragma once
#include "./perlin_noise.hpp"

namespace core::math::noise {

template <std::size_t N, typename T = float>
class BillowNoise : public PerlinNoise<N, T> {
 public:
  using VectorType = Vector<N, T>;

  using PerlinNoise<N, T>::PerlinNoise;

  T noise(const VectorType& point) const override {
    return std::abs(PerlinNoise<N, T>::noise(point)) * 2 - 1;
  }
};

}  // namespace core::math::noise