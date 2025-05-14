#pragma once

#include <fmt/core.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_group.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "./parallel/parallel_executor.hpp"  // Include your parallel utilities

namespace core::math::diffeq {

/**
 * @concept ArithmeticValue
 * @brief Concept for arithmetic types supporting basic operations
 *
 * Requires types to support:
 * - Standard arithmetic operations (+, -, *, /)
 * - Unary negation
 * - std::is_arithmetic_v trait
 */
template <typename T>
concept ArithmeticValue = requires([[maybe_unused]] T a, [[maybe_unused]] T b) {
  requires std::is_arithmetic_v<T>;
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a* b } -> std::same_as<T>;
  { a / b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
};

/**
 * @concept FloatingPoint
 * @brief Concept for floating-point types
 */
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/**
 * @concept SequenceContainer
 * @brief Concept for sequence containers with arithmetic value_type
 *
 * Requires containers to provide:
 * - size(), begin(), end() methods
 * - operator[] access
 * - value_type that satisfies ArithmeticValue
 */
template <typename T>
concept SequenceContainer = requires(T t) {
  typename T::value_type;
  t.size();
  t.begin();
  t.end();
  t[0];
  requires ArithmeticValue<typename T::value_type>;
};

namespace detail {

/**
 * @internal
 * @brief Creates a range sequence in a dynamic container
 *
 * @tparam Container Sequence container type
 * @param start Range start value (inclusive)
 * @param end Range end value (inclusive)
 * @param step Step size between values
 * @return Container Filled with sequence values
 * @throws std::invalid_argument if step is zero
 */
template <SequenceContainer Container>
constexpr Container create_range(typename Container::value_type start,
                                 typename Container::value_type end,
                                 typename Container::value_type step) {
  if (step == 0) {
    throw std::invalid_argument("Step size cannot be zero");
  }

  const auto size = static_cast<std::size_t>((end - start) / step) + 1;
  Container sequence{};
  sequence.reserve(size);

  for (std::size_t i = 0; i < size; ++i) {
    sequence.push_back(start + i * step);
  }

  return sequence;
}

/**
 * @internal
 * @brief Creates a range sequence in a fixed-size array
 *
 * @tparam ValueType Numeric type for values
 * @tparam N Array size
 * @param start Range start value (inclusive)
 * @param end Range end value (inclusive)
 * @param step Step size between values
 * @return std::array<ValueType, N> Filled with sequence values
 * @throws std::invalid_argument if step is zero
 */
template <FloatingPoint ValueType, std::size_t N>
constexpr std::array<ValueType, N> create_range(ValueType start, ValueType end,
                                                ValueType step) {
  if (step == 0) {
    throw std::invalid_argument("Step size cannot be zero");
  }

  std::array<ValueType, N> arr;
  for (std::size_t i = 0; i < N; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}
}  // namespace detail

/**
 * @class InitialValueProblem
 * @brief Represents an Initial Value Problem (IVP) for an ODE
 *
 * @tparam ValueType Floating-point type for calculations
 *
 * Models a differential equation of form:
 * \f[
 *   \frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0
 * \f]
 */
template <FloatingPoint ValueType>
class InitialValueProblem {
 public:
  /// Type alias for differential equation function
  using DifferentialFunction = std::function<ValueType(ValueType, ValueType)>;

  /**
   * @brief Constructs a new InitialValueProblem
   *
   * @param differential_equation Function f(t, y) defining the ODE
   * @param initial_time Initial time t0
   * @param initial_state Initial value y(t0)
   * @param description Optional problem description
   * @throws std::invalid_argument if differential_equation is not callable
   */
  InitialValueProblem(DifferentialFunction differential_equation,
                      ValueType initial_time, ValueType initial_state,
                      std::string_view description = "")
      : differential_equation_(std::move(differential_equation)),
        initial_time_(initial_time),
        initial_state_(initial_state),
        description_(description) {
    if (!differential_equation_) {
      throw std::invalid_argument(
          "Differential equation function must be callable");
    }
  }

  /**
   * @brief Evaluates the differential equation at given (t, y)
   *
   * @param time Time value t
   * @param state State value y
   * @return ValueType Result of f(t, y)
   */
  ValueType evaluate(ValueType time, ValueType state) const {
    return differential_equation_(time, state);
  }

  /// @return Initial time t0
  ValueType initial_time() const { return initial_time_; }

  /// @return Initial state y(t0)
  ValueType initial_state() const { return initial_state_; }

  /// @return Problem description
  std::string_view description() const { return description_; }

 private:
  DifferentialFunction differential_equation_;
  ValueType initial_time_;
  ValueType initial_state_;
  std::string_view description_;
};

/**
 * @class OdeSolver
 * @brief High-performance ODE solver with TBB parallelization support
 *
 * @tparam ValueType Floating-point type for calculations
 *
 * Supports multiple integration methods with both fixed and adaptive step
 * sizes. Parallel computation available for suitable methods via Intel TBB.
 */
template <FloatingPoint ValueType>
class OdeSolver {
 public:
  /// Available integration methods
  enum class IntegrationMethod {
    Euler,         /// Euler integration method
    Heun,          /// Modified Euler method
    RungeKutta4,   /// RungeKutta 4 solver
    AdaptiveRK45,  /// Adaptive Runge-Kutta-Fehlberg
    Verlet
  };

  /// Solution structure containing time points and states
  struct Solution {
    std::vector<ValueType> time_points;  ///< Time grid points
    std::vector<ValueType> states;       ///< Computed solution values
    std::vector<ValueType> errors;       ///< Error estimates (adaptive methods)
  };

  /**
   * @brief Constructs a new OdeSolver
   *
   * @param method Integration method to use (default: RungeKutta4)
   * @param parallelize Enable parallel computation (default: true)
   */
  explicit OdeSolver(IntegrationMethod method = IntegrationMethod::RungeKutta4,
                     bool parallelize = true)
      : method_(method), parallelize_(parallelize) {}

  /// Sets the integration method
  void set_method(IntegrationMethod method) { method_ = method; }

  /// @return Current integration method
  IntegrationMethod method() const { return method_; }

  /**
   * @brief Solves the IVP over a fixed time grid
   *
   * @param problem InitialValueProblem to solve
   * @param end_time Final time value
   * @param step_size Fixed step size
   * @return Solution Computed solution
   * @throws std::invalid_argument if step_size <= 0
   */
  Solution solve(const InitialValueProblem<ValueType>& problem,
                 ValueType end_time, ValueType step_size) const {
    if (step_size <= 0) {
      throw std::invalid_argument("Step size must be positive");
    }

    const auto time_points = detail::create_range<std::vector<ValueType>>(
        problem.initial_time(), end_time, step_size);

    Solution solution{};
    solution.time_points = time_points;
    solution.states.resize(time_points.size());
    solution.states[0] = problem.initial_state();

    if (parallelize_) {
      parallel_solve(problem, solution, step_size);
    } else {
      sequential_solve(problem, solution, step_size);
    }

    return solution;
  }

  /**
   * @brief Solves the IVP and returns solution at specific target time
   *
   * @param problem InitialValueProblem to solve
   * @param target_time Time point to solve for
   * @param initial_step Initial step size
   * @param tolerance Error tolerance for adaptive methods
   * @return ValueType Solution at target_time
   * @throws std::invalid_argument if step direction doesn't reach target
   */
  ValueType solve_at(const InitialValueProblem<ValueType>& problem,
                     ValueType target_time, ValueType initial_step,
                     [[maybe_unused]] ValueType tolerance = 1e-6) const {
    if ((target_time - problem.initial_time()) * initial_step <= 0) {
      throw std::invalid_argument("Step direction doesn't reach target");
    }

    ValueType current_time = problem.initial_time();
    ValueType current_state = problem.initial_state();
    ValueType step = initial_step;

    while (std::abs(current_time - target_time) > std::abs(step) * 0.5) {
      if (std::abs(current_time + step - target_time) >
          std::abs(target_time - current_time)) {
        step = target_time - current_time;
      }

      auto [new_state, error] =
          adaptive_step(problem, current_time, current_state, step, tolerance);

      current_state = new_state;
      current_time += step;

      // Adaptive step adjustment
      if (error > tolerance) {
        step *= 0.9 * std::sqrt(tolerance / error);
      } else {
        step *= 1.1;
      }
    }

    return current_state;
  }

  /**
   * @brief Performs Verlet integration for 2nd order ODEs
   *
   * Solves equations of form:
   * \f[
   *   \frac{d^2x}{dt^2} = a(x, v)
   * \f]
   *
   * @param acceleration_function Function a(x, v) computing acceleration
   * @param time_points Predefined time grid
   * @param initial_position Initial position x(t0)
   * @param initial_velocity Initial velocity v(t0)
   * @return Solution Contains positions (states) and velocities (errors)
   * @throws std::invalid_argument if time_points.size() < 2
   */
  static Solution verlet_integration(
      std::function<ValueType(ValueType, ValueType)> acceleration_function,
      const std::vector<ValueType>& time_points, ValueType initial_position,
      ValueType initial_velocity) {
    if (time_points.size() < 2) {
      throw std::invalid_argument("Need at least 2 time points");
    }

    const ValueType time_step = time_points[1] - time_points[0];
    Solution solution{};
    solution.time_points = time_points;
    solution.states.reserve(time_points.size());
    solution.errors.reserve(time_points.size());

    solution.states.push_back(initial_position);
    solution.errors.push_back(
        initial_velocity);  // Reusing errors for velocities

    ValueType previous_position =
        initial_position - initial_velocity * time_step;

    for (std::size_t i = 1; i < time_points.size(); ++i) {
      const ValueType current_position = solution.states[i - 1];
      const ValueType current_velocity = solution.errors[i - 1];
      const ValueType acceleration =
          acceleration_function(current_position, current_velocity);

      const ValueType new_position = 2 * current_position - previous_position +
                                     acceleration * time_step * time_step;
      const ValueType new_velocity =
          (new_position - previous_position) / (2 * time_step);

      solution.states.push_back(new_position);
      solution.errors.push_back(new_velocity);
      previous_position = current_position;
    }

    return solution;
  }

 private:
  IntegrationMethod method_;
  bool parallelize_;

  void sequential_solve(const InitialValueProblem<ValueType>& problem,
                        Solution& solution, ValueType step_size) const {
    switch (method_) {
      case IntegrationMethod::Euler:
        euler_method(problem, solution, step_size);
        break;
      case IntegrationMethod::Heun:
        heun_method(problem, solution, step_size);
        break;
      case IntegrationMethod::RungeKutta4:
        runge_kutta_4(problem, solution, step_size);
        break;
      case IntegrationMethod::AdaptiveRK45:
        adaptive_runge_kutta_fehlberg(problem, solution, step_size);
        break;
      default:
        throw std::runtime_error("Method not implemented for this signature");
    }
  }

  void parallel_solve(const InitialValueProblem<ValueType>& problem,
                      Solution& solution, ValueType step_size) const {
    // Only parallelize for methods that can be safely parallelized
    if (method_ == IntegrationMethod::RungeKutta4 ||
        method_ == IntegrationMethod::Euler ||
        method_ == IntegrationMethod::Heun) {
      core::math::parallel::parallel_for(
          size_t(1), solution.time_points.size(), [&](size_t i) {
            solution.states[i] =
                single_step(problem, solution.time_points[i - 1],
                            solution.states[i - 1], step_size);
          });
    } else {
      sequential_solve(problem, solution, step_size);
    }
  }

  ValueType single_step(const InitialValueProblem<ValueType>& problem,
                        ValueType current_time, ValueType current_state,
                        ValueType step_size) const {
    switch (method_) {
      case IntegrationMethod::Euler:
        return euler_step(problem, current_time, current_state, step_size);
      case IntegrationMethod::Heun:
        return heun_step(problem, current_time, current_state, step_size);
      case IntegrationMethod::RungeKutta4:
        return rk4_step(problem, current_time, current_state, step_size);
      default:
        throw std::runtime_error("Method not supported for parallel execution");
    }
  }

  // Individual step implementations
  static ValueType euler_step(const InitialValueProblem<ValueType>& problem,
                              ValueType time, ValueType state, ValueType step) {
    return state + step * problem.evaluate(time, state);
  }

  static ValueType heun_step(const InitialValueProblem<ValueType>& problem,
                             ValueType time, ValueType state, ValueType step) {
    const ValueType k1 = problem.evaluate(time, state);
    const ValueType k2 = problem.evaluate(time + step, state + step * k1);
    return state + (step / 2) * (k1 + k2);
  }

  static ValueType rk4_step(const InitialValueProblem<ValueType>& problem,
                            ValueType time, ValueType state, ValueType step) {
    const ValueType half_step = step / 2;
    const ValueType k1 = problem.evaluate(time, state);
    const ValueType k2 =
        problem.evaluate(time + half_step, state + half_step * k1);
    const ValueType k3 =
        problem.evaluate(time + half_step, state + half_step * k2);
    const ValueType k4 = problem.evaluate(time + step, state + step * k3);
    return state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
  }

  // Solution implementations
  static void euler_method(const InitialValueProblem<ValueType>& problem,
                           Solution& solution, ValueType step_size) {
    for (std::size_t i = 1; i < solution.time_points.size(); ++i) {
      solution.states[i] = euler_step(problem, solution.time_points[i - 1],
                                      solution.states[i - 1], step_size);
    }
  }

  static void heun_method(const InitialValueProblem<ValueType>& problem,
                          Solution& solution, ValueType step_size) {
    for (std::size_t i = 1; i < solution.time_points.size(); ++i) {
      solution.states[i] = heun_step(problem, solution.time_points[i - 1],
                                     solution.states[i - 1], step_size);
    }
  }

  static void runge_kutta_4(const InitialValueProblem<ValueType>& problem,
                            Solution& solution, ValueType step_size) {
    for (std::size_t i = 1; i < solution.time_points.size(); ++i) {
      solution.states[i] = rk4_step(problem, solution.time_points[i - 1],
                                    solution.states[i - 1], step_size);
    }
  }

  static void adaptive_runge_kutta_fehlberg(
      const InitialValueProblem<ValueType>& problem, Solution& solution,
      ValueType initial_step_size) {
    constexpr ValueType tolerance = 1e-6;
    constexpr ValueType safety_factor = 0.9;
    constexpr ValueType min_scale = 0.2;
    constexpr ValueType max_scale = 5.0;

    constexpr std::size_t max_steps = 100000;

    ValueType time = problem.initial_time();
    ValueType state = problem.initial_state();
    ValueType h = initial_step_size;

    solution.time_points.clear();
    solution.states.clear();
    solution.errors.clear();

    solution.time_points.push_back(time);
    solution.states.push_back(state);
    solution.errors.push_back(0);  // initial error is 0

    std::size_t steps = 0;

    while (time < solution.time_points.back() + h && steps++ < max_steps) {
      const ValueType k1 = h * problem.evaluate(time, state);
      const ValueType k2 =
          h * problem.evaluate(time + h * 0.25, state + k1 * 0.25);
      const ValueType k3 =
          h * problem.evaluate(time + h * 3.0 / 8.0,
                               state + 3.0 / 32.0 * k1 + 9.0 / 32.0 * k2);
      const ValueType k4 =
          h * problem.evaluate(time + h * 12.0 / 13.0,
                               state + 1932.0 / 2197.0 * k1 -
                                   7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3);
      const ValueType k5 =
          h * problem.evaluate(time + h, state + 439.0 / 216.0 * k1 - 8.0 * k2 +
                                             3680.0 / 513.0 * k3 -
                                             845.0 / 4104.0 * k4);
      const ValueType k6 =
          h * problem.evaluate(time + h / 2.0,
                               state - 8.0 / 27.0 * k1 + 2.0 * k2 -
                                   3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 -
                                   11.0 / 40.0 * k5);

      // 4th-order estimate
      const ValueType y4 = state + (25.0 / 216.0) * k1 +
                           (1408.0 / 2565.0) * k3 + (2197.0 / 4104.0) * k4 -
                           (1.0 / 5.0) * k5;

      // 5th-order estimate
      const ValueType y5 = state + (16.0 / 135.0) * k1 +
                           (6656.0 / 12825.0) * k3 + (28561.0 / 56430.0) * k4 -
                           (9.0 / 50.0) * k5 + (2.0 / 55.0) * k6;

      const ValueType error = std::abs(y5 - y4);

      // Adjust step size
      if (error <= tolerance) {
        // Accept step
        time += h;
        state = y4;

        solution.time_points.push_back(time);
        solution.states.push_back(state);
        solution.errors.push_back(error);
      }

      const ValueType scale =
          safety_factor * std::pow(tolerance / (error + 1e-10), 0.25);
      h *= std::clamp(scale, min_scale, max_scale);

      if (h < std::numeric_limits<ValueType>::epsilon()) {
        throw std::runtime_error("Step size became too small");
      }

      if (time + h > solution.time_points.back()) {
        h = solution.time_points.back() - time;
      }
    }

    if (steps >= max_steps) {
      throw std::runtime_error("Maximum number of steps exceeded");
    }
  }

  static std::pair<ValueType, ValueType> adaptive_step(
      const InitialValueProblem<ValueType>& problem, ValueType time,
      ValueType state, ValueType step, ValueType tolerance) {
    // Embedded Runge-Kutta method for error estimation
    const ValueType k1 = problem.evaluate(time, state);
    const ValueType k2 =
        problem.evaluate(time + step / 2, state + (step / 2) * k1);
    const ValueType k3 =
        problem.evaluate(time + step / 2, state + (step / 2) * k2);
    const ValueType k4 = problem.evaluate(time + step, state + step * k3);

    // 4th order solution
    const ValueType high_order_solution =
        state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4);

    // 2nd order solution for error estimation
    const ValueType low_order_solution = state + step * k2;

    const ValueType error = std::abs(high_order_solution - low_order_solution);
    const ValueType scaled_error = error / (std::abs(high_order_solution) +
                                            std::abs(step * k1) + tolerance);
    return {high_order_solution, scaled_error};
  }
};

/**
 * @brief Creates an InitialValueProblem with given parameters
 *
 * @tparam ValueType Floating point type for calculations
 * @param f Differential equation function (dy/dt = f(t, y))
 * @param initial_time Initial time value (t0)
 * @param initial_state Initial state value (y(t0))
 * @param description Optional description of the problem
 * @return InitialValueProblem<ValueType> Configured IVP object
 *
 * @example
 * auto problem = make_ivp<double>([](double t, double y) { return y; },
 * 0.0, 1.0);
 */
template <FloatingPoint ValueType>
auto make_ivp(typename InitialValueProblem<ValueType>::DifferentialFunction f,
              ValueType initial_time, ValueType initial_state,
              std::string_view description = "") {
  return InitialValueProblem<ValueType>(std::move(f), initial_time,
                                        initial_state, description);
}

/**
 * @brief Creates a uniformly spaced time range as a dynamic container
 *
 * @tparam Container Sequence container type (e.g. std::vector<double>)
 * @param start Start of the time range (inclusive)
 * @param end End of the time range (inclusive)
 * @param step Step size between points
 * @return Container Generated sequence container with time points
 *
 * @throws std::invalid_argument if step size is zero
 *
 * @note The container must support push_back() and reserve() methods
 *
 * @example
 * auto times = make_time_range<std::vector<double>>(0.0, 1.0, 0.1);
 */
template <SequenceContainer Container>
auto make_time_range(typename Container::value_type start,
                     typename Container::value_type end,
                     typename Container::value_type step) {
  return detail::create_range<Container>(start, end, step);
}

/**
 * @brief Creates a uniformly spaced time range as a fixed-size array
 *
 * @tparam ValueType Floating point type for time points
 * @tparam N Number of points in the array
 * @param start Start of the time range (inclusive)
 * @param end End of the time range (inclusive)
 * @param step Step size between points
 * @return std::array<ValueType, N> Generated array with time points
 *
 * @throws std::invalid_argument if step size is zero
 * @note The array size N must match the actual number of points
 *
 * @example
 * auto times = make_time_range<double, 11>(0.0, 1.0, 0.1);
 */
template <FloatingPoint ValueType, std::size_t N>
auto make_time_range(ValueType start, ValueType end, ValueType step) {
  return detail::create_range<std::array<ValueType, N>>(start, end, step);
}

}  // namespace core::math::diffeq
