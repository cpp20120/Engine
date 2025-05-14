
add_executable(${CMAKE_PROJECT_NAME} main.cpp)

#add_library(CoreMeta INTERFACE)

#target_include_directories(CoreMeta INTERFACE 
#    ${CMAKE_SOURCE_DIR}/lib/CoreMeta/include
#)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CoreMath)

if(WIN32 AND BUILD_SHARED_LIBS)
    # Альтернативный вариант (универсальный)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:CoreMath>
            $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>
            COMMENT "Copying DLL to executable directory"
    )
endif()