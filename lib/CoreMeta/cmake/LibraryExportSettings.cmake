include(GenerateExportHeader)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

if(TARGET CoreMeta)
    generate_export_header(CoreMeta
        BASE_NAME LIB
        EXPORT_MACRO_NAME LIB_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/lib_export.h"
        STATIC_DEFINE LIB_STATIC_DEFINE
    )
    install(FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/include/lib_export.h"
        DESTINATION include
    )

else()
    message(WARNING "Target 'CoreMeta' not found - export headers not generated")
endif()