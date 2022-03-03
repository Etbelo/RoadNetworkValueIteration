cmake_minimum_required (VERSION 3.0)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fopenmp")

set(LIB_NAME backend)

project(${LIB_NAME}.exe)

set(DEST_LIB ${PROJECT_SOURCE_DIR}/../backend/lib)

set(DEST_INC ${PROJECT_SOURCE_DIR}/../backend)  

add_subdirectory(lib)

include_directories(lib/inc)

set(SRC_FILES main.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} ${LIB_NAME})
