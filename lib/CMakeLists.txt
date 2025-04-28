# Добавляем библиотеки
add_subdirectory(CoreMath)
add_subdirectory(CoreMeta)

# Настройка тестов
option(BUILD_TESTS "Build tests" ON)
option(BUILD_ALL_TESTS "Build all tests automatically" ON)

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    if(BUILD_ALL_TESTS)
        # Добавляем тестовые поддиректории
        add_subdirectory(tests/CoreMath_test)
        add_subdirectory(tests/CoreMeta_test)

        # Общая цель для запуска всех тестов
        add_custom_target(run_all_libs_tests
                COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
                COMMENT "Running all libraries tests"
                VERBATIM
        )

        # Добавляем зависимости если цели существуют
        foreach(test_target IN ITEMS CoreMath_test CoreMeta_test)
            if(TARGET ${test_target})
                add_dependencies(run_all_libs_tests ${test_target})
            endif()
        endforeach()
    endif()
endif()