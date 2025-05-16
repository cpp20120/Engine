#pragma once
#include <array>
#include <numeric>

namespace core::math::noise {

template <typename T = float>
class PinkNoise {
 public:
  PinkNoise() : gen(std::random_device{}()) { reset(); }

  explicit PinkNoise(uint32_t seed) : gen(seed) { reset(); }

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

  void reset() {
    std::uniform_real_distribution<T> dist(-1, 1);
    for (auto& val : b) {
      val = dist(gen);
    }
  }

 private:
  std::array<T, 7> b;
  std::mt19937 gen;
};

}  // namespace core::math::noise