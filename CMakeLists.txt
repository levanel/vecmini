cmake_minimum_required(VERSION 3.22)
project(vecmini VERSION 1.0 LANGUAGES CXX)

# Use modern C++17 for cleaner syntax
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
)

# We will uncomment this later when we write the test script
add_executable(test_flat tests/test_flat_accuracy.cpp)
target_link_libraries(test_flat vectordb)