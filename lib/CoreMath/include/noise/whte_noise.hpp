#pragma once
#include <random>

namespace core::math::noise {

template <std::size_t N, typename T = float>
class WhiteNoise {
 public:
  using VectorType = Vector<N, T>;

  WhiteNoise() : gen(std::random_device{}()) {}
  explicit WhiteNoise(uint32_t seed) : gen(seed) {}

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
  mutable std::mt19937 gen;
};

}  // namespace core::math::noise