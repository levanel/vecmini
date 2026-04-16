cmake_minimum_required(VERSION 3.22)
project(vecmini VERSION 1.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-mavx2 -mfma -O3 -march=native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(vectordb STATIC 
    src/IndexFlat.cpp
    src/clustering.cpp
    src/IndexIVF.cpp
    src/IndexPQ.cpp
    src/IndexIVFPQ.cpp
)

add_executable(metadatatest tests/metadatatest.cpp)
target_link_libraries(metadatatest PRIVATE vectordb)

add_executable(test_flat tests/test_flat_accuracy.cpp)
target_link_libraries(test_flat vectordb)
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
#pybind11_add_module(vecmini src/bindings.cpp)
#   target_link_libraries(vecmini PRIVATE vectordb)
