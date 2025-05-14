#include <gtest/gtest.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "../../CoreMath/include/diffeq.hpp"
#include "../../CoreMath/include/matrix.hpp"
#include "diffeq.hpp"

using namespace core::math::diffeq;
using namespace core::math::matrix;

// Тест для InitialValueProblem
TEST(InitialValueProblemTest, ConstructorAndAccessors) {
  auto f = [](double t, double y) { return t + y; };
  InitialValueProblem<double> problem(f, 0.0, 1.0, "Test Problem");

  EXPECT_EQ(problem.initial_time(), 0.0);
  EXPECT_EQ(problem.initial_state(), 1.0);
  EXPECT_EQ(problem.description(), "Test Problem");
  EXPECT_EQ(problem.evaluate(1.0, 2.0), 3.0);
}

TEST(InitialValueProblemTest, InvalidFunctionThrows) {
  EXPECT_THROW(InitialValueProblem<double>(nullptr, 0.0, 1.0),
               std::invalid_argument);
}

// Тест для OdeSolver
TEST(OdeSolverTest, SolveWithEulerMethod) {
  auto f = [](double t, double y) { return y; };
  InitialValueProblem<double> problem(f, 0.0, 1.0);

  OdeSolver<double> solver(OdeSolver<double>::IntegrationMethod::Euler);
  auto solution = solver.solve(problem, 1.0, 0.1);

  EXPECT_EQ(solution.time_points.size(), 11);  // 0.0 to 1.0 with step 0.1
  EXPECT_NEAR(solution.states.back(), std::exp(1.0), 0.1);
}

TEST(OdeSolverTest, SolveWithRungeKutta4) {
  auto f = [](double t, double y) { return y; };
  InitialValueProblem<double> problem(f, 0.0, 1.0);

  OdeSolver<double> solver(OdeSolver<double>::IntegrationMethod::RungeKutta4);
  auto solution = solver.solve(problem, 1.0, 0.1);

  EXPECT_EQ(solution.time_points.size(), 11);  // 0.0 to 1.0 with step 0.1
  EXPECT_NEAR(solution.states.back(), std::exp(1.0), 0.01);
}

TEST(OdeSolverTest, AdaptiveRK45Method) {
  auto f = [](double t, double y) { return y; };
  InitialValueProblem<double> problem(f, 0.0, 1.0);

  OdeSolver<double> solver(OdeSolver<double>::IntegrationMethod::AdaptiveRK45);
  auto result = solver.solve_at(problem, 1.0, 0.1);

  EXPECT_NEAR(result, std::exp(1.0), 1e-6);
}

TEST(OdeSolverTest, InvalidStepSizeThrows) {
  auto f = [](double t, double y) { return y; };
  InitialValueProblem<double> problem(f, 0.0, 1.0);

  OdeSolver<double> solver;
  EXPECT_THROW(solver.solve(problem, 1.0, 0.0), std::invalid_argument);
}

TEST(OdeSolverTest, VerletIntegration) {
  auto acceleration = [](double x, double v) { return -x; };
  std::vector<double> time_points = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  double initial_position = 1.0;
  double initial_velocity = 0.0;

  auto solution = OdeSolver<double>::verlet_integration(
      acceleration, time_points, initial_position, initial_velocity);

  EXPECT_EQ(solution.time_points.size(), time_points.size());
  EXPECT_NEAR(solution.states.back(), std::cos(0.5),
              0.01);  // Проверка гармонического осциллятора
}

int runTests(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}