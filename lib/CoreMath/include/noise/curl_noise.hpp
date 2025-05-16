#pragma once
#include "./perlin_noise.hpp"

namespace core::math::noise {

template <std::size_t N, typename T = float>
class CurlNoise {
 public:
  static_assert(N == 2 || N == 3, "Curl noise only implemented for 2D and 3D");

  using VectorType = Vector<N, T>;

  CurlNoise(T scale = 1.0) : scale(scale) {}

  VectorType noise(const VectorType& point) const {
    if constexpr (N == 2) {
      return curl2D(point);
    } else {
      return curl3D(point);
    }
  }

 private:
  T scale;
  PerlinNoise<N, T> pn[4];  // Multiple noise functions for derivatives

  Vector<N, T> curl2D(const Vector<N, T>& p) const {
    const T eps = 0.0001;
    const Vector<N, T> p_x{eps, 0};
    const Vector<N, T> p_y{0, eps};

    // Get finite differences
    const T a = (pn[0].noise(p + p_y) - pn[0].noise(p - p_y)) / (2 * eps);
    const T b = (pn[1].noise(p + p_x) - pn[1].noise(p - p_x)) / (2 * eps);

    return Vector<N, T>{a - b, 0} * scale;
  }

  Vector<N, T> curl3D(const Vector<N, T>& p) const {
    const T eps = 0.0001;
    const Vector<N, T> p_x{eps, 0, 0};
    const Vector<N, T> p_y{0, eps, 0};
    const Vector<N, T> p_z{0, 0, eps};

    // Calculate Jacobian
    Vector<N, T> dF[3];
    for (int i = 0; i < 3; ++i) {
      dF[i][0] = (pn[i].noise(p + p_x) - pn[i].noise(p - p_x)) / (2 * eps);
      dF[i][1] = (pn[i].noise(p + p_y) - pn[i].noise(p - p_y)) / (2 * eps);
      dF[i][2] = (pn[i].noise(p + p_z) - pn[i].noise(p - p_z)) / (2 * eps);
    }

    // Return curl
    return Vector<N, T>{dF[2][1] - dF[1][2], dF[0][2] - dF[2][0],
                        dF[1][0] - dF[0][1]} *
           scale;
  }
};

}  // namespace core::math::noise