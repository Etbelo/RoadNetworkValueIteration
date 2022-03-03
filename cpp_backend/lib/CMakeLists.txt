cmake_minimum_required (VERSION 3.0)

set(VERBOSE ON)

if(NOT DEFINED LIB_NAME)
  set(LIB_NAME backend)
  set(DEST_LIB "${PROJECT_SOURCE_DIR}/install/lib")
  set(DEST_INC "${PROJECT_SOURCE_DIR}/install/")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/;${CMAKE_MODULE_PATH}")

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)    
find_package(LAPACK REQUIRED)  
find_package(LAPACKE REQUIRED)

set(SRC_FILES 
  ${CMAKE_CURRENT_LIST_DIR}/src/MDPGeneration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/MDPValueIteration.cpp
)

set(INC_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

if(VERBOSE)
  message(STATUS "Activate VERBOSE")
  add_definitions(-DVERBOSE)
endif()


include_directories(
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${LAPACKE_INCLUDE_DIRS_DEP}
)
                    
add_library(${LIB_NAME} SHARED ${SRC_FILES})

target_link_libraries(${LIB_NAME} OpenMP::OpenMP_CXX ${LAPACKE_LIBRARIES_DEP})

if(WIN32)
  message("No install step for windows!")
else()
  install(DIRECTORY ${INC_DIR} DESTINATION ${DEST_INC} MESSAGE_NEVER FILES_MATCHING PATTERN "*" )
  install(TARGETS ${LIB_NAME} LIBRARY DESTINATION ${DEST_LIB} CONFIGURATIONS Release)
endif()
