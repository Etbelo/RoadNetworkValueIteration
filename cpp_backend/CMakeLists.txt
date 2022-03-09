cmake_minimum_required (VERSION 3.0)

set(LIB_NAME cpp_backend)
project(${LIB_NAME})

# Cmake compile flags: Warnings, Optimizations
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fopenmp")

# Set build type for maximum performance
set(CMAKE_BUILD_TYPE Release)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define backend interface target location
set(DEST_LIB ${PROJECT_SOURCE_DIR}/../backend/lib)
set(DEST_INC ${PROJECT_SOURCE_DIR}/../backend)  

# Specify cmake directory
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/;${CMAKE_MODULE_PATH}")

# Include cmake modules
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)    
find_package(LAPACK REQUIRED)  
find_package(LAPACKE REQUIRED)

# Build cpp_backend library
set(INC_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

set(SRC_FILES 
  ${CMAKE_CURRENT_LIST_DIR}/src/MDPGeneration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/MDPValueIteration.cpp
)

if(VERBOSE)
  message(STATUS "Activate VERBOSE mode")
  add_definitions(-DVERBOSE)
endif()

include_directories(
  ${INC_DIR}
  ${LAPACKE_INCLUDE_DIRS_DEP}
)
                    
add_library(${LIB_NAME} SHARED ${SRC_FILES})

target_link_libraries(${LIB_NAME} OpenMP::OpenMP_CXX ${LAPACKE_LIBRARIES_DEP})

# Copy cpp_backend library to backend directory
install(DIRECTORY ${INC_DIR} DESTINATION ${DEST_INC} MESSAGE_NEVER FILES_MATCHING PATTERN "*" )
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION ${DEST_LIB} CONFIGURATIONS Release)
