#pragma once
#include <cmath>
#include <random>

namespace core::math::noise {

template <std::size_t N, typename T = float>
class GaborNoise {
 public:
  using VectorType = Vector<N, T>;

  GaborNoise(T kernel_size = 0.1, T frequency = 10.0)
      : kernel_size(kernel_size),
        frequency(frequency),
        gen(std::random_device{}()) {}

  explicit GaborNoise(uint32_t seed, T kernel_size = 0.1, T frequency = 10.0)
      : kernel_size(kernel_size), frequency(frequency), gen(seed) {}

  T noise(const VectorType& point) const {
    // Find cell
    std::array<int, N> cell;
    for (size_t i = 0; i < N; ++i) {
      cell[i] = static_cast<int>(std::floor(point[i] / kernel_size));
    }

    // Get kernel for this cell
    const auto& kernel = get_kernel(cell);

    // Evaluate all nearby kernels
    T result = 0;
    constexpr int radius = 2;  // Check 2 cells in each direction

    std::array<int, N> min_cell, max_cell;
    for (size_t i = 0; i < N; ++i) {
      min_cell[i] = cell[i] - radius;
      max_cell[i] = cell[i] + radius;
    }

    std::array<int, N> current_cell;
    for (size_t i = 0; i < std::pow(2 * radius + 1, N); ++i) {
      int temp = i;
      for (size_t j = 0; j < N; ++j) {
        current_cell[j] = min_cell[j] + (temp % (2 * radius + 1));
        temp /= (2 * radius + 1);
      }

      const auto& k = get_kernel(current_cell);
      VectorType delta = point - k.position;
      T dist = delta.magnitude();

      if (dist < 3 * kernel_size) {  // 3 sigma cutoff
        // Gabor function
        T g = std::exp(-dist * dist / (2 * kernel_size * kernel_size));
        T oscillation = std::cos(2 * M_PI * frequency * delta.dot(k.direction));
        result += g * oscillation * k.amplitude;
      }
    }

    return result;
  }

 private:
  struct Kernel {
    VectorType position;
    VectorType direction;
    T amplitude;
  };

  T kernel_size;
  T frequency;
  mutable std::mt19937 gen;
  mutable std::uniform_real_distribution<T> amp_dist{0.8, 1.0};

  using CellKey = std::array<int, N>;
  mutable std::unordered_map<CellKey, Kernel, CellHash> kernels;

  struct CellHash {
    size_t operator()(const CellKey& key) const {
      size_t hash = 0;
      for (auto val : key) {
        hash ^= std::hash<int>()(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  const Kernel& get_kernel(const CellKey& key) const {
    auto it = kernels.find(key);
    if (it != kernels.end()) {
      return it->second;
    }

    // Create new kernel
    Kernel k;
    for (size_t i = 0; i < N; ++i) {
      k.position[i] =
          (key[i] + std::uniform_real_distribution<T>{0, 1}(gen)) * kernel_size;
      k.direction[i] = std::uniform_real_distribution<T>{-1, 1}(gen);
    }
    k.direction = k.direction.normalize();
    k.amplitude = amp_dist(gen);

    kernels[key] = k;
    return kernels[key];
  }
};

}  // namespace core::math::noise