# lib/cmake/LibraryCompilerSettings.cmake
if(MSVC)
    target_compile_options(CoreMeta PRIVATE /W4 /WX)
else()
    target_compile_options(CoreMeta PRIVATE -Wall -Wextra)

    # Только для Linux/Unix
    if(UNIX AND NOT APPLE)
        target_compile_options(CoreMeta PRIVATE -fPIC)
    endif()
endif()

# Для shared библиотек
if(BUILD_SHARED_LIBS)
    target_compile_definitions(CoreMeta PRIVATE LIB_EXPORTS)
    if(NOT WIN32)
        target_compile_options(CoreMeta PRIVATE -fPIC)
    endif()
endif()