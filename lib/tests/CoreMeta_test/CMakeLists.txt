if(NOT TARGET CoreMeta)
    message(FATAL_ERROR "Library2 not found!")
endif()

if(NOT TARGET GTest::GTest)
    find_package(GTest REQUIRED)
endif()

add_executable(CoreMeta_test lib_test.cpp)
target_link_libraries(CoreMeta_test PRIVATE
        CoreMeta
        GTest::GTest
        GTest::Main
)

include(GoogleTest)
gtest_discover_tests(CoreMeta_test)