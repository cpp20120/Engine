#include <iostream>
#include <random>

#include "../include/diffeq.hpp"
#include "../include/matrix.hpp"
using namespace core::math::matrix;

auto equation = []([[maybe_unused]] double t, double y) { return y; };

auto test_problem = core::math::diffeq::make_ivp(equation, 0.0, 1.0, "y' = y");

core::math::diffeq::OdeSolver<double> solver(
    core::math::diffeq::OdeSolver<double>::IntegrationMethod::Euler);
core::math::diffeq::OdeSolver<double> solverRK(
    core::math::diffeq::OdeSolver<double>::IntegrationMethod::AdaptiveRK45);

auto test_solution = solver.solve(test_problem, 1.0, 0.1);
auto solutionRK = solverRK.solve(test_problem, 1.0, 0.1);

core::math::matrix::mat3x3 matA{1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                6.0f, 7.0f, 8.0f, 9.0f};
auto identity = core::math::matrix::mat3x3::identity();

auto matSum = matA + identity;

auto matScaled = matA * 2.5f;

core::math::matrix::mat3x3 matB = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f,
                                   4.0f, 3.0f, 2.0f, 1.0f};
auto matProduct = matA * matB;

auto matTransposed = core::math::matrix::transpose(matA);

float det = core::math::matrix::determinant(matA);
auto translate = core::math::matrix::transform::translate2d(10.0f, 5.0f);
auto rotate =
    core::math::matrix::transform::rotate2d(3.14159f / 4);  // 45 градусов
auto test_scale = core::math::matrix::transform::scale2d(2.0f, 3.0f);

auto transform2D = translate * rotate * test_scale;

auto translate3D = core::math::matrix::transform::translate3d(1.0f, 2.0f, 3.0f);
auto rotateX = core::math::matrix::transform::rotate3d_x(3.14159f / 6);
auto perspective = core::math::matrix::mat4x4::identity();

auto [L, U, P] = core::math::matrix::lu_decomposition(matA);

auto [Q, R] = core::math::matrix::qr_decomposition(matA);

auto submatrix = matA.slice<1, 3, 0, 2>();

auto row = core::math::matrix::get_row(matA, 1);
auto col = core::math::matrix::get_column(matA, 2);

// Матрица поворота на 90 градусов
auto rot90 = core::math::matrix::transform::rotate2d(3.14159f / 2);

// 1. Create 3x3 matrix with random floats in default range [0, 1]
// core::math::matrix::Matrix<float, 3, 3> mat1;
// core::math::matrix::randomize(mat1);

// 2. Double matrix with range [-5.0, 5.0]
// core::math::matrix::Matrix<double, 2, 2> mat2;
// core::math::matrix::randomize(mat2, -5.0, 5.0);

// 3. Float matrix with symmetric range [-10.0f, 10.0f]
// core::math::matrix::Matrix<float, 4, 4> mat3;
// core::math::matrix::randomize(mat3, 10.0f);

// 4. Integer matrix [1, 100]
// core::math::matrix::Matrix<int, 3, 3> mat4;
// core::math::matrix::randomize(mat4, 1, 100);

// 5. Custom generator
// std::mt19937 gen(42);
// core::math::matrix::Matrix<double, 2, 2> mat5;
// core::math::matrix::randomize(mat5, 0.0, 1.0, gen);
