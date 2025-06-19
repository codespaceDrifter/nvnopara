cmake_minimum_required(VERSION 3.10)

project(nvnopara, LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")


set (HEADERS
  ${PROJECT_ROOT_DIR}/inc
)
include_directories(${HEADERS})

file (GLOB_RECURSE TENSOR_SRC ${PROJECT_ROOT_DIR}/src/*.cpp)
file (GLOB_RECURSE CUDA_SRC ${PROJECT_ROOT_DIR}/src/*.cu)

set(SOURCES
  ${TENSOR_SRC}
  ${CUDA_SRC}
)

add_executable(main ${PROJECT_ROOT_DIR}/test/main.cpp ${SOURCES})
target_link_libraries(main PRIVATE cuda cudart)

add_executable(speed ${PROJECT_ROOT_DIR}/test/speed.cpp ${SOURCES})
target_link_libraries(speed PRIVATE cuda cudart)

add_executable(correct ${PROJECT_ROOT_DIR}/test/correct.cpp ${SOURCES})
target_link_libraries(correct PRIVATE cuda cudart)

#commands to build this:
# (in cmake folder)
#mkdir build
#cd build
#cmake -G Ninja ..
#ninja
