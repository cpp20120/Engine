#pragma once

#include <malloc.h>
#ifdef NOT WIN32
#include <cstdlib>
#endif  // NOT WIN32

namespace core::math::alloc {
/**
 * @brief Custom aligned allocator for cache-friendly memory allocation.
 * @tparam T Type of elements to allocate.
 * @tparam Alignment Memory alignment boundary.
 */
template <typename T, size_t Alignment = 64>
class AlignedAllocator {
 public:
  using value_type = T;

  /**
   * @brief Allocates memory with specified alignment.
   * @param n Number of elements to allocate.
   * @return Pointer to the allocated memory.
   * @throws std::bad_alloc If allocation fails.
   */
  T* allocate(size_t n) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
    ptr = std::aligned_alloc(Alignment, n * sizeof(T));
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  /**
   * @brief Deallocates memory.
   * @param ptr Pointer to the memory to deallocate.
   * @param size Size of the memory block (unused).
   */
  void deallocate(T* ptr, size_t) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    // Use std::free for non-Windows platforms
    std::free(ptr);
#endif
  }

  /**
   * @brief Rebind allocator to another type.
   * @tparam U New type to rebind to.
   */
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};
}  // namespace core::math::alloc