cmake_minimum_required(VERSION 3.15)
project(GradProject CXX)

# Force modern pure C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Point CMake directly to the vcpkg built package files
set(OpenCV_ROOT "${CMAKE_CURRENT_BINARY_DIR}/vcpkg_installed/x64-windows/share/opencv4")

# 1. Locate OpenCV (Provided by vcpkg)
find_package(OpenCV CONFIG REQUIRED)

# 2. Locate the local Python Environment 
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 3. Locate Pybind11
# Note: Ensure pybind11 is installed via vcpkg or pip in your environment
find_package(pybind11 REQUIRED)

# 4. Define the Core Engine Executable
add_executable(core_engine core_engine/src/main.cpp)

# 5. Link OpenCV and the Pybind11 Embed module to the Core Engine
target_link_libraries(core_engine PRIVATE ${OpenCV_LIBS} pybind11::embed)