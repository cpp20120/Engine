cmake_minimum_required(VERSION 3.26)
project(CoreMath LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#find_package(fmt)
find_package(TBB CONFIG REQUIRED)
find_package(mdspan REQUIRED)

add_subdirectory(src)
#target_link_libraries(CoreMath PRIVATE fmt::fmt)
target_link_libraries(CoreMath PRIVATE TBB::tbb)
target_link_libraries(CoreMath PRIVATE mdspan::mdspan)
# Версионирование
set_target_properties(CoreMath PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
)
include(${PROJECT_SOURCE_DIR}/cmake/FindHWlock.cmake)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(HWLOC IMPORTED_TARGET hwloc>=1.0)
endif()

if(WIN32 AND NOT HWLOC_FOUND)
  find_library(HWLOC_DLL "hwloc")
  find_library(HWLOC_LIB "libhwloc.lib")
  find_path(HWLOC_INCLUDE_DIRS "hwloc.h")
  if (HWLOC_LIB AND HWLOC_DLL AND HWLOC_INCLUDE_DIRS)
    set(HWLOC_FOUND 1)
  endif()
endif()

if(HWLOC_FOUND)
  set(DEFAULT_ENABLE_HWLOC ON CACHE INTERNAL "default hwloc")
else()
  set(DEFAULT_ENABLE_HWLOC OFF CACHE INTERNAL "default hwloc")
endif()

option(ENABLE_HWLOC "Enable Portable Hardware Locality software package"
        ${DEFAULT_ENABLE_HWLOC})
setup_cached_var(ENABLE_HWLOC "Using hwloc"
        "Requested build with hwloc, but no hwloc found!"
        "Hwloc found, but requested build without it")

include(sanitizers)

######################################################################################

if(ENABLE_TBB_DEVICE)
  # related documentation: https://github.com/oneapi-src/oneTBB/tree/tbb_2020/cmake#binary-package-integration
  find_package(TBB REQUIRED)
  if(TARGET TBB::tbb)
    set(TBB_IMPORTED_TARGETS TBB::tbb)
  elseif(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
    message(STATUS "searching for TBB with pkg-config")
    pkg_check_modules(TBB tbb>=2015)
    if(TBB_FOUND)
      set(TBB_IMPORTED_TARGETS "${TBB_LIBRARIES}" CACHE STRING "TBB libraries" FORCE)
    endif()
  endif()

  if(NOT TBB_INCLUDE_DIRS)
    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  if(NOT TBB_LIBRARIES)
    get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)
  endif()

  if(NOT TBB_FOUND)
    message(FATAL_ERROR "Can't find TBB libraries")
  endif()

  message(STATUS "Found TBB library: ${TBB_LIBRARIES} with include dirs: ${TBB_INCLUDE_DIRS}" )

  # The tbb device depends on the pthread device which depends on the basic device. ENABLE_HOST_CPU_DEVICES enables both.
  set(ENABLE_HOST_CPU_DEVICES 1)
endif()
#include(cmake/LibraryExportSettings.cmake)
include(cmake/LibraryCompilerSettings.cmake)
include(cmake/LibraryLinkageSettings.cmake)
#include(cmake/LibraryInstallSettings.cmake)

include(cmake/LibraryCP.cmake)


option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_TESTS "Build tests" OFF)

#configure_library_type(CoreMath_test)

#target_include_directories(CoreMath_test PUBLIC
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#        $<INSTALL_INTERFACE:include>
#)

include(cmake/LibraryInstallSettings.cmake)
