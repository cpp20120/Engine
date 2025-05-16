# lib/cmake/LibraryCompilerSettings.cmake
if(MSVC)
    target_compile_options(CoreUI PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(CoreUI PRIVATE -Wall -Wextra)

    # Только для Linux/Unix
    if(UNIX AND NOT APPLE)
        target_compile_options(CoreUI PRIVATE -fPIC)
    endif()
endif()

# Для shared библиотек
if(BUILD_SHARED_LIBS)
    target_compile_definitions(CoreUI PRIVATE LIB_EXPORTS)
    if(NOT WIN32)
        target_compile_options(CoreUI PRIVATE -fPIC)
    endif()
endif()