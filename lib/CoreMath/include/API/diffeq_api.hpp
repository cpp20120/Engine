#pragma once

#include "../diffeq.hpp"

namespace core::math::diffeq::api {

/**
 * @class DifferentialEquationsAPI
 * @brief High-level API for solving differential equations.
 */
class DifferentialEquationsAPI {
 public:
  // Solver Creation

  /**
   * @brief Creates a solver with the specified integration method.
   * @tparam ValueType Floating point type for calculations.
   * @param method Integration method to use.
   * @param parallelize Enable parallel computation.
   * @return OdeSolver with the specified method.
   */
  template <FloatingPoint ValueType>
  static auto create_solver(typename core::math::diffeq::OdeSolver<
                                ValueType>::IntegrationMethod method,
                            bool parallelize = true) {
    return core::math::diffeq::OdeSolver<ValueType>(method, parallelize);
  }

  // Initial Value Problem Creation

  /**
   * @brief Creates an Initial Value Problem (IVP) with the given parameters.
   * @tparam ValueType Floating point type for calculations.
   * @param f Differential equation function (dy/dt = f(t, y)).
   * @param initial_time Initial time value (t0).
   * @param initial_state Initial state value (y(t0)).
   * @param description Optional description of the problem.
   * @return InitialValueProblem with the given parameters.
   */
  template <FloatingPoint ValueType>
  static auto create_ivp(typename core::math::diffeq::InitialValueProblem<
                             ValueType>::DifferentialFunction f,
                         ValueType initial_time, ValueType initial_state,
                         std::string_view description = "") {
    return core::math::diffeq::make_ivp(std::move(f), initial_time,
                                        initial_state, description);
  }

  // Time Range Creation

  /**
   * @brief Creates a uniformly spaced time range as a dynamic container.
   * @tparam Container Sequence container type (e.g., std::vector<double>).
   * @param start Start of the time range (inclusive).
   * @param end End of the time range (inclusive).
   * @param step Step size between points.
   * @return Container Generated sequence container with time points.
   * @throws std::invalid_argument if step size is zero.
   */
  template <SequenceContainer Container>
  static auto create_time_range(typename Container::value_type start,
                                typename Container::value_type end,
                                typename Container::value_type step) {
    return core::math::diffeq::make_time_range<Container>(start, end, step);
  }

  /**
   * @brief Creates a uniformly spaced time range as a fixed-size array.
   * @tparam ValueType Floating point type for time points.
   * @tparam N Number of points in the array.
   * @param start Start of the time range (inclusive).
   * @param end End of the time range (inclusive).
   * @param step Step size between points.
   * @return std::array<ValueType, N> Generated array with time points.
   * @throws std::invalid_argument if step size is zero.
   */
  template <FloatingPoint ValueType, std::size_t N>
  static auto create_time_range(ValueType start, ValueType end,
                                ValueType step) {
    return core::math::diffeq::make_time_range<ValueType, N>(start, end, step);
  }

  // Solving the IVP

  /**
   * @brief Solves the IVP over a fixed time grid.
   * @tparam ValueType Floating point type for calculations.
   * @param solver OdeSolver to use.
   * @param problem InitialValueProblem to solve.
   * @param end_time Final time value.
   * @param step_size Fixed step size.
   * @return Solution Computed solution.
   * @throws std::invalid_argument if step_size <= 0.
   */
  template <FloatingPoint ValueType>
  static auto solve(
      core::math::diffeq::OdeSolver<ValueType>& solver,
      const core::math::diffeq::InitialValueProblem<ValueType>& problem,
      ValueType end_time, ValueType step_size) {
    return solver.solve(problem, end_time, step_size);
  }

  /**
   * @brief Solves the IVP and returns the solution at a specific target time.
   * @tparam ValueType Floating point type for calculations.
   * @param solver OdeSolver to use.
   * @param problem InitialValueProblem to solve.
   * @param target_time Time point to solve for.
   * @param initial_step Initial step size.
   * @param tolerance Error tolerance for adaptive methods.
   * @return ValueType Solution at target_time.
   * @throws std::invalid_argument if step direction doesn't reach target.
   */
  template <FloatingPoint ValueType>
  static auto solve_at(
      core::math::diffeq::OdeSolver<ValueType>& solver,
      const core::math::diffeq::InitialValueProblem<ValueType>& problem,
      ValueType target_time, ValueType initial_step,
      ValueType tolerance = 1e-6) {
    return solver.solve_at(problem, target_time, initial_step, tolerance);
  }

  /**
   * @brief Performs Verlet integration for 2nd order ODEs.
   * @tparam ValueType Floating point type for calculations.
   * @param acceleration_function Function a(x, v) computing acceleration.
   * @param time_points Predefined time grid.
   * @param initial_position Initial position x(t0).
   * @param initial_velocity Initial velocity v(t0).
   * @return Solution Contains positions (states) and velocities (errors).
   * @throws std::invalid_argument if time_points.size() < 2.
   */
  template <FloatingPoint ValueType>
  static auto verlet_integration(
      std::function<ValueType(ValueType, ValueType)> acceleration_function,
      const std::vector<ValueType>& time_points, ValueType initial_position,
      ValueType initial_velocity) {
    return core::math::diffeq::OdeSolver<ValueType>::verlet_integration(
        acceleration_function, time_points, initial_position, initial_velocity);
  }
};

}  // namespace core::math::diffeq::api
