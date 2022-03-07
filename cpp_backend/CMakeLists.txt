cmake_minimum_required (VERSION 3.0)

set(LIB_NAME backend)
project(${LIB_NAME}_test)

# Cmake compile flags: Warnings, Optimizations
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fopenmp")

# Define backend interface target location
set(DEST_LIB ${PROJECT_SOURCE_DIR}/../backend/lib)
set(DEST_INC ${PROJECT_SOURCE_DIR}/../backend)  

# Add cpp_backend library
add_subdirectory(lib)

# Build cpp_backend test function
include_directories(lib/inc)

set(SRC_FILES main.cpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} ${LIB_NAME})
