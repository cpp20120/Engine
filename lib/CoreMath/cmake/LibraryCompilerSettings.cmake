# lib/cmake/LibraryCompilerSettings.cmake
if(MSVC)
    target_compile_options(CoreMath PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(CoreMath PRIVATE -Wall -Wextra)

    # Только для Linux/Unix
    if(UNIX AND NOT APPLE)
        target_compile_options(CoreMath PRIVATE -fPIC)
    endif()
endif()

# Для shared библиотек
if(BUILD_SHARED_LIBS)
    target_compile_definitions(CoreMath PRIVATE LIB_EXPORTS)
    if(NOT WIN32)
        target_compile_options(CoreMath PRIVATE -fPIC)
    endif()
endif()