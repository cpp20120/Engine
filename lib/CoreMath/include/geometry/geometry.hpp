#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <optional>
#include <print>
#include <stdexcept>
#include <tuple>
#include <vector>

/**
 * @namespace core::math::geometry
 * @brief Namespace for core mathematical geometry classes and functions.
 */
namespace core::math::geometry {

// Forward declarations
template <typename T>
class Point;
template <typename T>
class Line;
template <typename T>
class Plane;
template <typename T>
class Circle;
template <typename T>
class Sphere;
template <typename T>
class Ellipsoid;
template <typename T>
class AABB;
template <typename T>
class Rectangle;
template <typename T>
class Capsule;
template <typename T>
class Cone;
template <typename T>
class Cylinder;
template <typename T>
class CatmullRomSpline;
template <typename T>
class BSpline;

/**
 * @class Point
 * @brief Represents a point in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Point {
 public:
  T x, y, z;

  /**
   * @brief Constructs a Point with given coordinates.
   * @param x The x-coordinate.
   * @param y The y-coordinate.
   * @param z The z-coordinate.
   */
  constexpr Point(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}

  /**
   * @brief Adds two points.
   * @param other The other point to add.
   * @return The resulting point.
   */
  constexpr Point operator+(const Point& other) const {
    return {x + other.x, y + other.y, z + other.z};
  }

  /**
   * @brief Subtracts two points.
   * @param other The other point to subtract.
   * @return The resulting point.
   */
  constexpr Point operator-(const Point& other) const {
    return {x - other.x, y - other.y, z - other.z};
  }

  /**
   * @brief Multiplies the point by a scalar.
   * @param scalar The scalar to multiply by.
   * @return The resulting point.
   */
  constexpr Point operator*(T scalar) const {
    return {x * scalar, y * scalar, z * scalar};
  }

  /**
   * @brief Divides the point by a scalar.
   * @param scalar The scalar to divide by.
   * @return The resulting point.
   */
  constexpr Point operator/(T scalar) const {
    static_assert(
        !std::is_floating_point_v<T> || std::numeric_limits<T>::is_iec559,
        "T must support IEEE 754 division.");
    return {x / scalar, y / scalar, z / scalar};
  }

  /**
   * @brief Computes the dot product of two points.
   * @param other The other point.
   * @return The dot product.
   */
  constexpr T dot(const Point& other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  /**
   * @brief Computes the cross product of two points.
   * @param other The other point.
   * @return The resulting point.
   */
  constexpr Point cross(const Point& other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z,
            x * other.y - y * other.x};
  }

  /**
   * @brief Computes the squared magnitude of the point.
   * @return The squared magnitude.
   */
  constexpr T magnitudeSquared() const { return dot(*this); }

  /**
   * @brief Computes the magnitude of the point.
   * @return The magnitude.
   */
  constexpr T magnitude() const { return std::sqrt(magnitudeSquared()); }

  /**
   * @brief Normalizes the point.
   * @return The normalized point.
   */
  constexpr Point normalized() const {
    constexpr T epsilon =
        std::is_floating_point_v<T> ? static_cast<T>(1e-9) : T(1);
    T mag = magnitude();
    static_assert(
        !std::is_floating_point_v<T> || std::numeric_limits<T>::is_iec559,
        "T must support IEEE 754 floating-point arithmetic for normalization.");
    return *this / (mag > epsilon ? mag : static_cast<T>(1));
  }

  /**
   * @brief Checks if two points are equal.
   * @param other The other point.
   * @return True if the points are equal, false otherwise.
   */
  constexpr bool operator==(const Point& other) const {
    constexpr T epsilon =
        std::is_floating_point_v<T> ? static_cast<T>(1e-9) : T(0);
    return std::abs(x - other.x) < epsilon && std::abs(y - other.y) < epsilon &&
           std::abs(z - other.z) < epsilon;
  }

  /**
   * @brief Checks if two points are not equal.
   * @param other The other point.
   * @return True if the points are not equal, false otherwise.
   */
  constexpr bool operator!=(const Point& other) const {
    return !(*this == other);
  }

  /**
   * @brief Computes the squared distance between two points.
   * @param other The other point.
   * @return The squared distance.
   */
  constexpr T distanceSquared(const Point& other) const {
    return (*this - other).magnitudeSquared();
  }

  /**
   * @brief Computes the distance between two points.
   * @param other The other point.
   * @return The distance.
   */
  constexpr T distance(const Point& other) const {
    return (*this - other).magnitude();
  }

  /**
   * @brief Accesses the coordinate at the given index.
   * @param index The index of the coordinate.
   * @return The coordinate.
   */
  constexpr T operator[](size_t index) const {
    switch (index) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw std::out_of_range("Point index out of range");
    }
  }

  /**
   * @brief Accesses the coordinate at the given index.
   * @param index The index of the coordinate.
   * @return The coordinate.
   */
  constexpr T& operator[](size_t index) {
    switch (index) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw std::out_of_range("Point index out of range");
    }
  }
};

/**
 * @class Line
 * @brief Represents a line in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Line {
 public:
  Point<T> origin;
  Point<T> direction;

  /**
   * @brief Constructs a Line with given origin and direction.
   * @param origin The origin of the line.
   * @param direction The direction of the line.
   */
  constexpr Line(const Point<T>& origin, const Point<T>& direction)
      : origin(origin), direction(direction.normalized()) {}

  /**
   * @brief Computes the intersection of the line with a plane.
   * @param plane The plane to intersect with.
   * @return The intersection point, if any.
   */
  constexpr std::optional<Point<T>> intersect(const Plane<T>& plane) const {
    T denom = plane.normal.dot(direction);
    if (std::abs(denom) < std::numeric_limits<T>::epsilon())
      return std::nullopt;

    T t = -(plane.normal.dot(origin) + plane.d) / denom;
    return origin + direction * t;
  }

  /**
   * @brief Computes the intersection of the line with a sphere.
   * @param sphere The sphere to intersect with.
   * @return The intersection points, if any.
   */
  constexpr std::optional<std::pair<Point<T>, Point<T>>> intersect(
      const Sphere<T>& sphere) const {
    Point<T> oc = origin - sphere.center;
    T a = direction.dot(direction);
    T b = 2 * oc.dot(direction);
    T c = oc.dot(oc) - sphere.radius * sphere.radius;
    T discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return std::nullopt;

    T sqrt_disc = std::sqrt(discriminant);
    T t1 = (-b - sqrt_disc) / (2 * a);
    T t2 = (-b + sqrt_disc) / (2 * a);

    return std::make_pair(origin + direction * t1, origin + direction * t2);
  }

  /**
   * @brief Computes the intersection of the line with an ellipsoid.
   * @param ellipsoid The ellipsoid to intersect with.
   * @return The intersection points, if any.
   */
  constexpr std::optional<std::pair<Point<T>, Point<T>>> intersect(
      const Ellipsoid<T>& ellipsoid) const {
    Point<T> oc = origin - ellipsoid.center;
    Point<T> scaled_dir = {
        direction.x / (ellipsoid.radii.x * ellipsoid.radii.x),
        direction.y / (ellipsoid.radii.y * ellipsoid.radii.y),
        direction.z / (ellipsoid.radii.z * ellipsoid.radii.z)};
    Point<T> scaled_oc = {oc.x / (ellipsoid.radii.x * ellipsoid.radii.x),
                          oc.y / (ellipsoid.radii.y * ellipsoid.radii.y),
                          oc.z / (ellipsoid.radii.z * ellipsoid.radii.z)};

    T a = direction.dot(scaled_dir);
    T b = 2 * oc.dot(scaled_dir);
    T c = oc.dot(scaled_oc) - 1;
    T discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return std::nullopt;

    T sqrt_disc = std::sqrt(discriminant);
    T t1 = (-b - sqrt_disc) / (2 * a);
    T t2 = (-b + sqrt_disc) / (2 * a);

    return std::make_pair(origin + direction * t1, origin + direction * t2);
  }

  /**
   * @brief Computes the intersection of the line with a Catmull-Rom spline.
   * @param spline The Catmull-Rom spline to intersect with.
   * @return The intersection point and parameter, if any.
   */
  constexpr std::optional<std::pair<T, Point<T>>> intersect(
      const CatmullRomSpline<T>& spline) const {
    return spline.intersect(*this);
  }

  /**
   * @brief Computes the intersection of the line with a B-spline.
   * @param spline The B-spline to intersect with.
   * @return The intersection point and parameter, if any.
   */
  constexpr std::optional<std::pair<T, Point<T>>> intersect(
      const BSpline<T>& spline) const {
    return spline.intersect(*this);
  }

  /**
   * @brief Projects a point onto the line.
   * @param point The point to project.
   * @return The projected point.
   */
  constexpr Point<T> projectPoint(const Point<T>& point) const {
    Point<T> diff = point - origin;
    T t = diff.dot(direction) / direction.dot(direction);
    return origin + direction * t;
  }
};

/**
 * @class Plane
 * @brief Represents a plane in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Plane {
 public:
  Point<T> normal;
  T d;

  /**
   * @brief Constructs a Plane with given normal and distance.
   * @param normal The normal vector of the plane.
   * @param d The distance from the origin.
   */
  constexpr Plane(const Point<T>& normal, T d)
      : normal(normal.normalized()), d(d) {}

  /**
   * @brief Constructs a Plane from three points.
   * @param p1 The first point.
   * @param p2 The second point.
   * @param p3 The third point.
   */
  constexpr Plane(const Point<T>& p1, const Point<T>& p2, const Point<T>& p3) {
    normal = (p2 - p1).cross(p3 - p1).normalized();
    d = -normal.dot(p1);
  }

  /**
   * @brief Computes the distance from a point to the plane.
   * @param point The point.
   * @return The distance.
   */
  constexpr T distanceTo(const Point<T>& point) const {
    return normal.dot(point) + d;
  }

  /**
   * @brief Checks if the plane contains a point.
   * @param point The point.
   * @return True if the plane contains the point, false otherwise.
   */
  constexpr bool contains(const Point<T>& point) const {
    return std::abs(distanceTo(point)) < std::numeric_limits<T>::epsilon();
  }

  /**
   * @brief Computes the intersection of the plane with another plane.
   * @param other The other plane.
   * @return The intersection line, if any.
   */
  constexpr std::optional<Line<T>> intersect(const Plane<T>& other) const {
    Point<T> dir = normal.cross(other.normal);
    if (dir.magnitude() < std::numeric_limits<T>::epsilon())
      return std::nullopt;

    Point<T> point =
        (other.normal * d - normal * other.d).cross(dir) / dir.dot(dir);
    return Line<T>(point, dir.normalized());
  }
};

/**
 * @class Circle
 * @brief Represents a circle in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Circle {
 public:
  Point<T> center;
  Point<T> normal;
  T radius;

  /**
   * @brief Constructs a Circle with given center, normal, and radius.
   * @param center The center of the circle.
   * @param normal The normal vector of the circle.
   * @param radius The radius of the circle.
   */
  constexpr Circle(const Point<T>& center, const Point<T>& normal, T radius)
      : center(center), normal(normal.normalized()), radius(radius) {
    if (radius < 0) throw std::runtime_error("Negative radius");
  }
};

/**
 * @class Sphere
 * @brief Represents a sphere in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Sphere {
 public:
  Point<T> center;
  T radius;

  /**
   * @brief Constructs a Sphere with given center and radius.
   * @param center The center of the sphere.
   * @param radius The radius of the sphere.
   */
  constexpr Sphere(const Point<T>& center, T radius)
      : center(center), radius(radius) {
    if (radius < 0) throw std::runtime_error("Negative radius");
  }

  /**
   * @brief Checks if the sphere intersects with an AABB.
   * @param aabb The AABB to check.
   * @return True if the sphere intersects with the AABB, false otherwise.
   */
  constexpr bool intersects(const AABB<T>& aabb) const {
    T dist_squared = 0;
    for (int i = 0; i < 3; ++i) {
      if (center[i] < aabb.min[i]) {
        T diff = center[i] - aabb.min[i];
        dist_squared += diff * diff;
      } else if (center[i] > aabb.max[i]) {
        T diff = center[i] - aabb.max[i];
        dist_squared += diff * diff;
      }
    }
    return dist_squared <= (radius * radius);
  }

  /**
   * @brief Computes the intersection of the sphere with another sphere.
   * @param other The other sphere.
   * @return The intersection circle, if any.
   */
  constexpr std::optional<Circle<T>> intersect(const Sphere<T>& other) const {
    Point<T> delta = other.center - center;
    T dist = delta.magnitude();

    if (dist < std::numeric_limits<T>::epsilon() ||
        dist > radius + other.radius ||
        dist < std::abs(radius - other.radius)) {
      return std::nullopt;
    }

    T a = (radius * radius - other.radius * other.radius + dist * dist) /
          (2 * dist);
    T h = std::sqrt(radius * radius - a * a);

    Point<T> circle_center = center + delta * (a / dist);
    return Circle<T>(circle_center, delta.normalized(), h);
  }
};

/**
 * @class Ellipsoid
 * @brief Represents an ellipsoid in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Ellipsoid {
 public:
  Point<T> center;
  Point<T> radii;

  /**
   * @brief Constructs an Ellipsoid with given center and radii.
   * @param center The center of the ellipsoid.
   * @param radii The radii of the ellipsoid.
   */
  constexpr Ellipsoid(const Point<T>& center, const Point<T>& radii)
      : center(center), radii(radii) {
    if (radii.x < 0 || radii.y < 0 || radii.z < 0)
      throw std::runtime_error("Negative radii");
  }
};

/**
 * @class AABB
 * @brief Represents an axis-aligned bounding box in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class AABB {
 public:
  Point<T> min;
  Point<T> max;

  /**
   * @brief Constructs an AABB with given minimum and maximum points.
   * @param min The minimum point.
   * @param max The maximum point.
   */
  constexpr AABB(const Point<T>& min, const Point<T>& max)
      : min(min), max(max) {
    if (min.x > max.x || min.y > max.y || min.z > max.z)
      throw std::runtime_error("Invalid AABB bounds");
  }

  /**
   * @brief Checks if the AABB intersects with another AABB.
   * @param other The other AABB.
   * @return True if the AABBs intersect, false otherwise.
   */
  constexpr bool intersects(const AABB& other) const {
    return (min.x <= other.max.x && max.x >= other.min.x) &&
           (min.y <= other.max.y && max.y >= other.min.y) &&
           (min.z <= other.max.z && max.z >= other.min.z);
  }

  /**
   * @brief Computes the intersection of the AABB with another AABB.
   * @param other The other AABB.
   * @return The intersection AABB, if any.
   */
  constexpr std::optional<AABB> intersection(const AABB& other) const {
    if (!intersects(other)) return std::nullopt;

    Point<T> new_min{std::max(min.x, other.min.x), std::max(min.y, other.min.y),
                     std::max(min.z, other.min.z)};

    Point<T> new_max{std::min(max.x, other.max.x), std::min(max.y, other.max.y),
                     std::min(max.z, other.max.z)};

    return AABB(new_min, new_max);
  }
};

/**
 * @class Rectangle
 * @brief Represents a rectangle in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Rectangle {
 public:
  Point<T> origin;
  Point<T> width_dir;
  Point<T> height_dir;
  T width;
  T height;

  /**
   * @brief Constructs a Rectangle with given origin, width direction, height
   * direction, width, and height.
   * @param origin The origin of the rectangle.
   * @param width_dir The width direction of the rectangle.
   * @param height_dir The height direction of the rectangle.
   * @param width The width of the rectangle.
   * @param height The height of the rectangle.
   */
  constexpr Rectangle(const Point<T>& origin, const Point<T>& width_dir,
                      const Point<T>& height_dir, T width, T height)
      : origin(origin),
        width_dir(width_dir.normalized()),
        height_dir(height_dir.normalized()),
        width(width),
        height(height) {
    if (width <= 0 || height <= 0)
      throw std::runtime_error("Width and height must be positive");

    if (std::abs(width_dir.dot(height_dir)) > std::numeric_limits<T>::epsilon())
      throw std::runtime_error(
          "Width and height directions must be orthogonal");
  }

  /**
   * @brief Computes the normal vector of the rectangle.
   * @return The normal vector.
   */
  constexpr Point<T> normal() const {
    return width_dir.cross(height_dir).normalized();
  }

  /**
   * @brief Computes the point at the given UV coordinates.
   * @param u The U coordinate.
   * @param v The V coordinate.
   * @return The point at the given UV coordinates.
   */
  constexpr Point<T> at(T u, T v) const {
    return origin + width_dir * (u * width) + height_dir * (v * height);
  }

  /**
   * @brief Checks if the rectangle contains a point.
   * @param point The point.
   * @return True if the rectangle contains the point, false otherwise.
   */
  constexpr bool contains(const Point<T>& point) const {
    Point<T> vec = point - origin;
    T u = vec.dot(width_dir) / width;
    T v = vec.dot(height_dir) / height;
    return u >= 0 && u <= 1 && v >= 0 && v <= 1;
  }

  /**
   * @brief Computes the intersection of the rectangle with a line.
   * @param line The line to intersect with.
   * @return The intersection point, if any.
   */
  constexpr std::optional<Point<T>> intersect(const Line<T>& line) const {
    Plane<T> plane(origin, normal());
    auto intersection = line.intersect(plane);
    if (!intersection) return std::nullopt;

    Point<T> pt = *intersection;
    Point<T> vec = pt - origin;
    T u = vec.dot(width_dir) / width;
    T v = vec.dot(height_dir) / height;

    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
      return pt;
    }
    return std::nullopt;
  }
};

/**
 * @class Capsule
 * @brief Represents a capsule in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Capsule {
 public:
  Point<T> start;
  Point<T> end;
  T radius;

  /**
   * @brief Constructs a Capsule with given start, end, and radius.
   * @param start The start point of the capsule.
   * @param end The end point of the capsule.
   * @param radius The radius of the capsule.
   */
  constexpr Capsule(const Point<T>& start, const Point<T>& end, T radius)
      : start(start), end(end), radius(radius) {
    if (radius < 0) throw std::runtime_error("Negative radius");
  }

  /**
   * @brief Checks if the capsule intersects with a sphere.
   * @param sphere The sphere to check.
   * @return True if the capsule intersects with the sphere, false otherwise.
   */
  constexpr bool intersects(const Sphere<T>& sphere) const {
    Line<T> line(start, end - start);
    auto projection = line.projectPoint(sphere.center);
    T dist = projection.distance(sphere.center);
    return dist <= (radius + sphere.radius);
  }
};

/**
 * @class Cone
 * @brief Represents a cone in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Cone {
 public:
  Point<T> apex;
  Point<T> direction;
  T height;
  T radius;

  /**
   * @brief Constructs a Cone with given apex, direction, height, and radius.
   * @param apex The apex of the cone.
   * @param direction The direction of the cone.
   * @param height The height of the cone.
   * @param radius The radius of the cone.
   */
  constexpr Cone(const Point<T>& apex, const Point<T>& direction, T height,
                 T radius)
      : apex(apex),
        direction(direction.normalized()),
        height(height),
        radius(radius) {
    if (height < 0 || radius < 0)
      throw std::runtime_error("Height and radius must be positive");
  }
};

/**
 * @class Cylinder
 * @brief Represents a cylinder in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class Cylinder {
 public:
  Point<T> start;
  Point<T> end;
  T radius;

  /**
   * @brief Constructs a Cylinder with given start, end, and radius.
   * @param start The start point of the cylinder.
   * @param end The end point of the cylinder.
   * @param radius The radius of the cylinder.
   */
  constexpr Cylinder(const Point<T>& start, const Point<T>& end, T radius)
      : start(start), end(end), radius(radius) {
    if (radius < 0) throw std::runtime_error("Negative radius");
  }

  /**
   * @brief Checks if the cylinder intersects with a sphere.
   * @param sphere The sphere to check.
   * @return True if the cylinder intersects with the sphere, false otherwise.
   */
  constexpr bool intersects(const Sphere<T>& sphere) const {
    Line<T> line(start, end - start);
    auto projection = line.projectPoint(sphere.center);
    T dist = projection.distance(sphere.center);
    return dist <= (radius + sphere.radius);
  }
};

/**
 * @class CatmullRomSpline
 * @brief Represents a Catmull-Rom spline in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class CatmullRomSpline {
 public:
  std::vector<Point<T>> controlPoints;
  bool closed;
  size_t segmentCount;
  T segmentSize;

  /**
   * @brief Constructs a Catmull-Rom spline with given control points and closed
   * flag.
   * @param points The control points of the spline.
   * @param closed Whether the spline is closed.
   */
  constexpr CatmullRomSpline(const std::vector<Point<T>>& points,
                             bool closed = false)
      : controlPoints(points), closed(closed) {
    if (controlPoints.size() < 4) {
      throw std::runtime_error(
          "Catmull-Rom spline requires at least 4 control points");
    }

    segmentCount = closed ? controlPoints.size() : controlPoints.size() - 3;
    segmentSize = T(1) / static_cast<T>(segmentCount);
  }

  /**
   * @brief Evaluates the spline at the given parameter.
   * @param t The parameter.
   * @return The point at the given parameter.
   */
  constexpr Point<T> evaluate(T t) const {
    if (controlPoints.empty()) return Point<T>{};

    t = std::clamp(t, T(0), T(1));

    if (t == T(0)) return controlPoints[closed ? 0 : 1];
    if (t == T(1))
      return closed ? controlPoints[0]
                    : controlPoints[controlPoints.size() - 2];

    size_t segment = static_cast<size_t>(t / segmentSize);
    t = (t - segment * segmentSize) / segmentSize;

    if (!closed && segment >= controlPoints.size() - 3) {
      segment = controlPoints.size() - 4;
      t = T(1);
    }

    const size_t i0 = closed ? segment % controlPoints.size() : segment;
    const size_t i1 = closed ? (i0 + 1) % controlPoints.size() : i0 + 1;
    const size_t i2 = closed ? (i0 + 2) % controlPoints.size() : i0 + 2;
    const size_t i3 = closed ? (i0 + 3) % controlPoints.size() : i0 + 3;

    const T t2 = t * t;
    const T t3 = t2 * t;

    const T b0 = (T(2) * t3 - T(3) * t2 + T(1));
    const T b1 = (T(-2) * t3 + T(3) * t2);
    const T b2 = (t3 - T(2) * t2 + t) * T(0.5);
    const T b3 = (t3 - t2) * T(0.5);

    return controlPoints[i1] * b0 + controlPoints[i2] * b1 +
           (controlPoints[i2] - controlPoints[i0]) * b2 +
           (controlPoints[i3] - controlPoints[i1]) * b3;
  }

  /**
   * @brief Computes the intersection of the spline with a line.
   * @param line The line to intersect with.
   * @param epsilon The tolerance for intersection.
   * @return The intersection parameter and point, if any.
   */
  constexpr std::optional<std::pair<T, Point<T>>> intersect(
      const Line<T>& line,
      T epsilon = std::numeric_limits<T>::epsilon()) const {
    const size_t initialSteps = 20;
    std::vector<std::pair<T, T>> candidates;

    for (size_t i = 0; i < initialSteps; ++i) {
      const T t0 = T(i) / T(initialSteps);
      const T t1 = T(i + 1) / T(initialSteps);

      const Point<T> p0 = evaluate(t0);
      const Point<T> p1 = evaluate(t1);

      if (line.distanceTo(p0) <= epsilon || line.distanceTo(p1) <= epsilon) {
        candidates.emplace_back(t0, t1);
        continue;
      }

      const Line<T> segment(p0, p1 - p0);
      if (line.intersect(segment).has_value()) {
        candidates.emplace_back(t0, t1);
      }
    }

    for (const auto& [t0, t1] : candidates) {
      const size_t refineSteps = 10;
      for (size_t j = 0; j < refineSteps; ++j) {
        const T rt0 = t0 + (t1 - t0) * T(j) / T(refineSteps);
        const T rt1 = t0 + (t1 - t0) * T(j + 1) / T(refineSteps);

        const Point<T> p0 = evaluate(rt0);
        const Point<T> p1 = evaluate(rt1);
        const Line<T> segment(p0, p1 - p0);

        const auto intersection = line.intersect(segment);
        if (intersection) {
          const Point<T> pt = *intersection;
          if ((pt - p0).magnitude() + (pt - p1).magnitude() <=
              (p1 - p0).magnitude() + epsilon) {
            return std::make_pair(rt0 + (rt1 - rt0) * (pt - p0).magnitude() /
                                            (p1 - p0).magnitude(),
                                  pt);
          }
        }
      }
    }

    return std::nullopt;
  }
};

/**
 * @class BSpline
 * @brief Represents a B-spline in 3D space.
 * @tparam T The type of the coordinates (e.g., float, double).
 */
template <typename T>
class BSpline {
 public:
  std::vector<Point<T>> controlPoints;
  size_t degree;
  std::vector<T> knots;

  /**
   * @brief Constructs a B-spline with given control points and degree.
   * @param points The control points of the spline.
   * @param degree The degree of the spline.
   */
  constexpr BSpline(const std::vector<Point<T>>& points, size_t degree = 3)
      : controlPoints(points), degree(degree) {
    if (controlPoints.size() < degree + 1) {
      throw std::runtime_error(
          "BSpline requires at least degree+1 control points");
    }
    generateKnotVector();
  }

  /**
   * @brief Evaluates the spline at the given parameter.
   * @param t The parameter.
   * @return The point at the given parameter.
   */
  constexpr Point<T> evaluate(T t) const {
    t = std::clamp(t, T(0), T(1));
    const T tScaled =
        t * (knots[controlPoints.size()] - knots[degree]) + knots[degree];

    Point<T> result{};
    for (size_t i = 0; i < controlPoints.size(); ++i) {
      const T basis = basisFunction(i, degree, tScaled);
      result = result + controlPoints[i] * basis;
    }
    return result;
  }

  /**
   * @brief Computes the intersection of the spline with a line.
   * @param line The line to intersect with.
   * @param epsilon The tolerance for intersection.
   * @return The intersection parameter and point, if any.
   */
  constexpr std::optional<std::pair<T, Point<T>>> intersect(
      const Line<T>& line,
      T epsilon = std::numeric_limits<T>::epsilon()) const {
    const size_t steps = 100;
    for (size_t i = 0; i < steps; ++i) {
      const T t0 = T(i) / T(steps);
      const T t1 = T(i + 1) / T(steps);

      const Point<T> p0 = evaluate(t0);
      const Point<T> p1 = evaluate(t1);

      const Line<T> segment(p0, p1 - p0);
      const auto intersection = line.intersect(segment);
      if (intersection) {
        const Point<T> pt = *intersection;
        if ((pt - p0).magnitude() + (pt - p1).magnitude() <=
            (p1 - p0).magnitude() + epsilon) {
          return std::make_pair(
              t0 + (t1 - t0) * (pt - p0).magnitude() / (p1 - p0).magnitude(),
              pt);
        }
      }
    }
    return std::nullopt;
  }

 private:
  /**
   * @brief Generates the knot vector for the spline.
   */
  constexpr void generateKnotVector() {
    knots.resize(controlPoints.size() + degree + 1);
    for (size_t i = 0; i < knots.size(); ++i) {
      knots[i] = static_cast<T>(i);
    }
  }

  /**
   * @brief Computes the basis function for the spline.
   * @param i The index of the control point.
   * @param k The degree of the spline.
   * @param t The parameter.
   * @return The basis function value.
   */
  constexpr T basisFunction(size_t i, size_t k, T t) const {
    if (k == 0) {
      return (t >= knots[i] && t < knots[i + 1]) ? T(1) : T(0);
    }

    T denom1 = knots[i + k] - knots[i];
    T term1 = (denom1 > T(0))
                  ? ((t - knots[i]) / denom1) * basisFunction(i, k - 1, t)
                  : T(0);

    T denom2 = knots[i + k + 1] - knots[i + 1];
    T term2 = (denom2 > T(0)) ? ((knots[i + k + 1] - t) / denom2) *
                                    basisFunction(i + 1, k - 1, t)
                              : T(0);

    return term1 + term2;
  }
};

}  // namespace core::math::geometry
