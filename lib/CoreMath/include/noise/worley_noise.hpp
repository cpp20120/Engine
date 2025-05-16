#pragma once
#include <algorithm>
#include <array>

#include "./noise_utils.hpp"

namespace core::math::noise {

template <std::size_t N, typename T = float, size_t K = 1>
class WorleyNoise {
 public:
  using VectorType = Vector<N, T>;

  WorleyNoise() { init_feature_points(); }
  explicit WorleyNoise(uint32_t seed) : gen(seed) { init_feature_points(); }

  T noise(const VectorType& point) const {
    std::array<T, K> distances;
    distances.fill(std::numeric_limits<T>::max());

    // Check neighboring cells
    std::array<int, N> cell;
    for (size_t i = 0; i < N; ++i) {
      cell[i] = static_cast<int>(std::floor(point[i] / cell_size));
    }

    constexpr int radius = 1;  // Check adjacent cells
    std::array<int, N> min_cell, max_cell;
    for (size_t i = 0; i < N; ++i) {
      min_cell[i] = cell[i] - radius;
      max_cell[i] = cell[i] + radius;
    }

    // Iterate through neighboring cells
    std::array<int, N> current_cell;
    for (size_t i = 0; i < std::pow(2 * radius + 1, N); ++i) {
      // Generate all combinations of cell coordinates
      int temp = i;
      for (size_t j = 0; j < N; ++j) {
        current_cell[j] = min_cell[j] + (temp % (2 * radius + 1));
        temp /= (2 * radius + 1);
      }

      // Get feature points for this cell
      const auto& points = get_cell_points(current_cell);
      for (const auto& fp : points) {
        T dist = (point - fp).magnitude();

        // Insert sorted
        for (size_t k = 0; k < K; ++k) {
          if (dist < distances[k]) {
            for (size_t l = K - 1; l > k; --l) {
              distances[l] = distances[l - 1];
            }
            distances[k] = dist;
            break;
          }
        }
      }
    }

    return distances[K - 1];  // Return K-th closest distance
  }

 private:
  static constexpr T cell_size = T(1.0);
  static constexpr size_t points_per_cell = 3;

  std::mt19937 gen;
  std::uniform_real_distribution<T> dist{0, cell_size};

  using CellKey = std::array<int, N>;
  std::unordered_map<CellKey, std::array<VectorType, points_per_cell>, CellHash>
      feature_points;

  struct CellHash {
    size_t operator()(const CellKey& key) const {
      size_t hash = 0;
      for (auto val : key) {
        hash ^= std::hash<int>()(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  void init_feature_points() {
    // In a real implementation, you'd generate these on demand
    // This is simplified for demonstration
  }

  const std::array<VectorType, points_per_cell>& get_cell_points(
      const CellKey& key) const {
    auto it = feature_points.find(key);
    if (it != feature_points.end()) {
      return it->second;
    }

    // Generate new feature points for this cell
    std::array<VectorType, points_per_cell> points;
    for (auto& p : points) {
      for (size_t i = 0; i < N; ++i) {
        p[i] = key[i] * cell_size + dist(gen);
      }
    }

    feature_points[key] = points;
    return feature_points[key];
  }
};

}  // namespace core::math::noise