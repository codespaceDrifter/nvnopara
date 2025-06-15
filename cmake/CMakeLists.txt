cmake_minimum_required(VERSION 3.10)

project(nvnopara, LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


set (HEADERS
  ${PROJECT_ROOT_DIR}/inc
)
include_directories(${HEADERS})

file (GLOB_RECURSE TENSOR_SRC ${PROJECT_ROOT_DIR}/src/*.cpp)


set(SOURCES
  ${TENSOR_SRC}
)



add_executable(main ${SOURCES})

target_link_libraries(main PRIVATE cuda cudart)


#commands to build this:
# (in cmake folder)
#mkdir build
#cd build
#cmake -G Ninja ..
#ninja
