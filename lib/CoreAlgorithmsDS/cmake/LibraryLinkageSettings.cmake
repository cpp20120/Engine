

include(cmake/LibraryCompilerSettings.cmake)
include(cmake/LibraryExportSettings.cmake)

function(configure_library_type TARGET_NAME)
    message(STATUS "Configuring library type for ${TARGET_NAME}")

    if(BUILD_SHARED_LIBS)
        set_target_properties(${TARGET_NAME} PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                OUTPUT_NAME "${TARGET_NAME}_shared"
        )

        if(WIN32)
            set_target_properties(${TARGET_NAME} PROPERTIES
                    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
                    SUFFIX ".dll"
                    PREFIX ""
            )
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES
                    SUFFIX ".so"
                    PREFIX "lib"
            )
        endif()
    else()
        set_target_properties(${TARGET_NAME} PROPERTIES
                OUTPUT_NAME "${TARGET_NAME}_static"
        )

        if(WIN32)
            set_target_properties(${TARGET_NAME} PROPERTIES
                    SUFFIX ".lib"
                    PREFIX ""
            )
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES
                    SUFFIX ".a"
                    PREFIX "lib"
            )
        endif()
    endif()
endfunction()

configure_library_type(CoreAlgorithmsDS)

set(LIB_VERSION 1.0.0)
set_target_properties(CoreAlgorithmsDS PROPERTIES
        VERSION ${LIB_VERSION}
        SOVERSION 1
)

target_include_directories(CoreAlgorithmsDS PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 10. Установка
#include(cmake/LibraryInstallSettings.cmake)