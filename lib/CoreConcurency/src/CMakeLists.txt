add_library(CoreMeta
  lib.cpp 
)

target_include_directories(CoreMeta PUBLIC include)
#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE lib2)
  