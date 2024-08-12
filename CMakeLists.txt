cmake_minimum_required(VERSION 3.20)
project(softmax LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11) # 设置CUDA标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # 设置CUDA标准是否必须

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.cuh")
message(STATUS "SOURCES: ${SOURCES}")
message(STATUS "HEADERS: ${HEADERS}")

add_executable(softmax ${SOURCES} ${HEADERS})
target_include_directories(softmax PRIVATE ${CMAKE_SOURCE_DIR}/include)