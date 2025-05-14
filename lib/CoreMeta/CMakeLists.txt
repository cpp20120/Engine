cmake_minimum_required(VERSION 3.26)
project(CoreMeta LANGUAGES CXX)

# Set C++ standard requirements
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create interface library for header-only
add_library(CoreMeta INTERFACE)

# Add include directories (interface target needs INTERFACE specifier)
target_include_directories(CoreMeta INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Versioning for the interface library
set_target_properties(CoreMeta PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# Optional: Add compiler settings if needed
include(cmake/LibraryCompilerSettings.cmake)

# Optional: Add CP (Code Protection?) settings if needed
include(cmake/LibraryCP.cmake)

# Installation configuration for header-only library
#install(
#    DIRECTORY include/
#    DESTINATION include
#    FILES_MATCHING PATTERN "*.hpp"
#)

# Optional: Export targets for find_package support
#install(TARGETS CoreMeta EXPORT CoreMetaTargets)
#install(EXPORT CoreMetaTargets
#    DESTINATION lib/cmake/CoreMeta
#    NAMESPACE CoreMeta::
#)