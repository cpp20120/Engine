if(NOT TARGET CoreMath)
    message(FATAL_ERROR "Library1 not found!")
endif()

if(NOT TARGET GTest::GTest)
    find_package(GTest REQUIRED)
endif()

add_executable(CoreMath_test lib_test.cpp)
target_link_libraries(CoreMath_test PRIVATE
        CoreMath
        GTest::GTest
        GTest::Main
)

include(GoogleTest)
gtest_discover_tests(CoreMath_test)