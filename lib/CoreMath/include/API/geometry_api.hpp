#pragma once

#include "../geometry/geometry.hpp"
namespace core::math::geometry::api {

/**
 * @class GeometryAPI
 * @brief High-level API for geometric operations
 */
class GeometryAPI {
 public:
  // Point operations

  /**
   * @brief Creates a point with given coordinates
   * @tparam T Arithmetic type of point coordinates
   * @param x X coordinate
   * @param y Y coordinate
   * @param z Z coordinate (default 0)
   * @return Point with specified coordinates
   */
  template <typename T>
  static auto create_point(T x, T y, T z = T{0}) {
    return Point<T>(x, y, z);
  }

  /**
   * @brief Adds two points
   * @tparam T Arithmetic type of points
   * @param a First point
   * @param b Second point
   * @return Sum of the points
   */
  template <typename T>
  static auto add(const Point<T>& a, const Point<T>& b) {
    return a + b;
  }

  /**
   * @brief Subtracts two points
   * @tparam T Arithmetic type of points
   * @param a First point
   * @param b Second point
   * @return Difference of the points
   */
  template <typename T>
  static auto subtract(const Point<T>& a, const Point<T>& b) {
    return a - b;
  }

  /**
   * @brief Computes dot product of two points
   * @tparam T Arithmetic type of points
   * @param a First point
   * @param b Second point
   * @return Dot product
   */
  template <typename T>
  static auto dot(const Point<T>& a, const Point<T>& b) {
    return a.dot(b);
  }

  /**
   * @brief Computes cross product of two points
   * @tparam T Arithmetic type of points
   * @param a First point
   * @param b Second point
   * @return Cross product
   */
  template <typename T>
  static auto cross(const Point<T>& a, const Point<T>& b) {
    return a.cross(b);
  }

  /**
   * @brief Computes distance between two points
   * @tparam T Arithmetic type of points
   * @param a First point
   * @param b Second point
   * @return Distance between points
   */
  template <typename T>
  static auto distance(const Point<T>& a, const Point<T>& b) {
    return a.distance(b);
  }

  // Line operations

  /**
   * @brief Creates a line from origin and direction
   * @tparam T Floating point type
   * @param origin Origin point
   * @param direction Direction vector
   * @return Line object
   */
  template <typename T>
  static auto create_line(const Point<T>& origin, const Point<T>& direction) {
    return Line<T>(origin, direction);
  }

  /**
   * @brief Finds intersection between line and plane
   * @tparam T Floating point type
   * @param line The line
   * @param plane The plane
   * @return Optional intersection point if exists
   */
  template <typename T>
  static auto intersect(const Line<T>& line, const Plane<T>& plane) {
    return line.intersect(plane);
  }

  /**
   * @brief Projects a point onto a line
   * @tparam T Floating point type
   * @param line The line
   * @param point The point to project
   * @return Projected point
   */
  template <typename T>
  static auto project_point(const Line<T>& line, const Point<T>& point) {
    return line.projectPoint(point);
  }

  // Plane operations

  /**
   * @brief Creates a plane from normal and distance
   * @tparam T Floating point type
   * @param normal Normal vector
   * @param d Distance from origin
   * @return Plane object
   */
  template <typename T>
  static auto create_plane(const Point<T>& normal, T d) {
    return Plane<T>(normal, d);
  }

  /**
   * @brief Creates a plane from three points
   * @tparam T Floating point type
   * @param p1 First point
   * @param p2 Second point
   * @param p3 Third point
   * @return Plane object
   */
  template <typename T>
  static auto create_plane_from_points(const Point<T>& p1, const Point<T>& p2,
                                       const Point<T>& p3) {
    return Plane<T>(p1, p2, p3);
  }

  /**
   * @brief Computes distance from point to plane
   * @tparam T Floating point type
   * @param plane The plane
   * @param point The point
   * @return Distance from point to plane
   */
  template <typename T>
  static auto distance_to(const Plane<T>& plane, const Point<T>& point) {
    return plane.distanceTo(point);
  }

  // Circle operations

  /**
   * @brief Creates a circle
   * @tparam T Floating point type
   * @param center Center point
   * @param normal Normal vector
   * @param radius Radius
   * @return Circle object
   */
  template <typename T>
  static auto create_circle(const Point<T>& center, const Point<T>& normal,
                            T radius) {
    return Circle<T>(center, normal, radius);
  }

  // Sphere operations

  /**
   * @brief Creates a sphere
   * @tparam T Floating point type
   * @param center Center point
   * @param radius Radius
   * @return Sphere object
   */
  template <typename T>
  static auto create_sphere(const Point<T>& center, T radius) {
    return Sphere<T>(center, radius);
  }

  /**
   * @brief Checks if sphere intersects with AABB
   * @tparam T Floating point type
   * @param sphere The sphere
   * @param aabb The axis-aligned bounding box
   * @return True if intersects, false otherwise
   */
  template <typename T>
  static auto intersects(const Sphere<T>& sphere, const AABB<T>& aabb) {
    return sphere.intersects(aabb);
  }

  // AABB operations

  /**
   * @brief Creates an axis-aligned bounding box
   * @tparam T Arithmetic type
   * @param min Minimum point
   * @param max Maximum point
   * @return AABB object
   */
  template <typename T>
  static auto create_aabb(const Point<T>& min, const Point<T>& max) {
    return AABB<T>(min, max);
  }

  /**
   * @brief Checks if two AABBs intersect
   * @tparam T Arithmetic type
   * @param a First AABB
   * @param b Second AABB
   * @return True if intersects, false otherwise
   */
  template <typename T>
  static auto intersects(const AABB<T>& a, const AABB<T>& b) {
    return a.intersects(b);
  }

  // Rectangle operations

  /**
   * @brief Creates a rectangle in 3D space
   * @tparam T Floating point type
   * @param origin Origin point
   * @param width_dir Width direction
   * @param height_dir Height direction
   * @param width Width
   * @param height Height
   * @return Rectangle object
   */
  template <typename T>
  static auto create_rectangle(const Point<T>& origin,
                               const Point<T>& width_dir,
                               const Point<T>& height_dir, T width, T height) {
    return Rectangle<T>(origin, width_dir, height_dir, width, height);
  }

  // Capsule operations

  /**
   * @brief Creates a capsule
   * @tparam T Floating point type
   * @param start Start point
   * @param end End point
   * @param radius Radius
   * @return Capsule object
   */
  template <typename T>
  static auto create_capsule(const Point<T>& start, const Point<T>& end,
                             T radius) {
    return Capsule<T>(start, end, radius);
  }

  // Cone operations

  /**
   * @brief Creates a cone
   * @tparam T Floating point type
   * @param apex Apex point
   * @param direction Direction vector
   * @param height Height
   * @param radius Radius
   * @return Cone object
   */
  template <typename T>
  static auto create_cone(const Point<T>& apex, const Point<T>& direction,
                          T height, T radius) {
    return Cone<T>(apex, direction, height, radius);
  }

  // Cylinder operations

  /**
   * @brief Creates a cylinder
   * @tparam T Floating point type
   * @param start Start point
   * @param end End point
   * @param radius Radius
   * @return Cylinder object
   */
  template <typename T>
  static auto create_cylinder(const Point<T>& start, const Point<T>& end,
                              T radius) {
    return Cylinder<T>(start, end, radius);
  }

  // Spline operations

  /**
   * @brief Creates a Catmull-Rom spline
   * @tparam T Floating point type
   * @param control_points Control points
   * @param closed Whether the spline is closed
   * @return CatmullRomSpline object
   */
  template <typename T>
  static auto create_catmull_rom_spline(
      const std::vector<Point<T>>& control_points, bool closed = false) {
    return CatmullRomSpline<T>(control_points, closed);
  }

  /**
   * @brief Creates a B-spline
   * @tparam T Floating point type
   * @param control_points Control points
   * @param degree Degree of the spline
   * @return BSpline object
   */
  template <typename T>
  static auto create_bspline(const std::vector<Point<T>>& control_points,
                             size_t degree = 3) {
    return BSpline<T>(control_points, degree);
  }
};

}  // namespace core::math::geometry::api