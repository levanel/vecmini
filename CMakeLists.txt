cmake_minimum_required(VERSION 3.22)
project(vecmini VERSION 1.0 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Use modern C++17 for cleaner syntax
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-mavx2 -mfma -O3 -march=native)
# THE CLANGD BRIDGE: This generates the compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tell the compiler where our header files live
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# PHASE 1: Build the core vector database library
# As we add more files (IndexIVF.cpp, etc.), we just add them to this list
add_library(vectordb STATIC 
    src/IndexFlat.cpp
    src/clustering.cpp
    src/IndexIVF.cpp
    src/IndexPQ.cpp
    src/IndexIVFPQ.cpp
)

# We will uncomment this later when we wr
add_executable(metadatatest tests/metadatatest.cpp)
target_link_libraries(metadatatest PRIVATE vectordb)

add_executable(test_flat tests/test_flat_accuracy.cpp)
target_link_libraries(test_flat vectordb)
# ==========================================
# PHASE 2: The Python Bridge (Pybind11)
# ==========================================

# 1. Find the Python interpreter and Pybind11 in our new venv
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)

# 2. Build the Python Shared Object (.so) module named 'vecmini'
# IMPORTANT: Replace 'src/bindings.cpp' with the actual name of your wrapper file!
#pybind11_add_module(vecmini src/bindings.cpp)

# 3. Link our core C++ engine (vectordb) into the Python module
#   target_link_libraries(vecmini PRIVATE vectordb)