cmake_minimum_required(VERSION 3.26)
project(CoreUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_subdirectory(src)

target_include_directories(CoreUI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${glad_INCLUDE_DIRS}
    ${IMGUI_INCLUDE_DIRS}
)
target_link_libraries(CoreUI PRIVATE imgui::imgui)
target_link_libraries(CoreUI PRIVATE glfw)
target_link_libraries(CoerUI PRIVATE glad::glad)

target_compile_definitions(CoreUI PRIVATE
    GLFW_INCLUDE_NONE
)
