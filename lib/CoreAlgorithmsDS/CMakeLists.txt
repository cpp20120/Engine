cmake_minimum_required(VERSION 3.26)
project(CoreAlgorithmsDS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#find_package(fmt)
#find_package(TBB CONFIG REQUIRED)
#find_package(mdspan REQUIRED)

add_subdirectory(src)
#target_link_libraries(CoreAlgorithmsDS PRIVATE fmt::fmt)
#target_link_libraries(CoreAlgorithmsDS PRIVATE TBB::tbb)
#target_link_libraries(CoreAlgorithmsDS PRIVATE mdspan::mdspan)
# Версионирование
set_target_properties(CoreAlgorithmsDS PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
)

#include(cmake/LibraryExportSettings.cmake)
include(cmake/LibraryCompilerSettings.cmake)
include(cmake/LibraryLinkageSettings.cmake)
#include(cmake/LibraryInstallSettings.cmake)

include(cmake/LibraryCP.cmake)


option(BUILD_SHARED_LIBS "Build as shared library" OFF)
option(BUILD_TESTS "Build tests" OFF)

#configure_library_type(CoreAlgorithmsDS_test)

#target_include_directories(CoreAlgorithmsDS_test PUBLIC
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#        $<INSTALL_INTERFACE:include>
#)

include(cmake/LibraryInstallSettings.cmake)
