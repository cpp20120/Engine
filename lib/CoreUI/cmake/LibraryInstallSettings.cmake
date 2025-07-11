# lib/cmake/LibraryInstallSettings.cmake

# Установка самой библиотеки и заголовков
install(TARGETS CoreUI
    EXPORT libTargets
    ARCHIVE DESTINATION CoreUI
    LIBRARY DESTINATION CoreUI
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lib_export.h" DESTINATION include)

# Генерация конфигурационных файлов для пакета
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libConfig.cmake"
    INSTALL_DESTINATION lib/cmake/CoreUI
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)


install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libConfigVersion.cmake"
    DESTINATION lib/cmake/CoreUI
)
