#pragma once
#include "./noise_utils.hpp"

namespace core::math::noise {

template <typename NoiseFn, std::size_t N, typename T = float>
class DomainWarpedNoise {
 public:
  using VectorType = Vector<N, T>;

  DomainWarpedNoise(const NoiseFn& noise, T amplitude = 1.0)
      : noise(noise), amplitude(amplitude) {}

  T noise(const VectorType& point) const {
    VectorType offset;
    for (size_t i = 0; i < N; ++i) {
      offset[i] = amplitude * noise(point);
    }
    return noise(point + offset);
  }

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

 private:
  NoiseFn noise;
  T amplitude;
};

}  // namespace core::math::noise