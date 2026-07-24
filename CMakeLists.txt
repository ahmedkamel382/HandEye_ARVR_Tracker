cmake_minimum_required(VERSION 3.15)
project(GradProject CXX)

# Force modern C++ standards for Engineer 1 and Engineer 4
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Locate OpenCV (Provided by vcpkg)
find_package(OpenCV CONFIG REQUIRED)

# 2. Locate the local Python Environment (Guided by CMakePresets)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 3. Define the Core Engine Executable
# Pointing to the specific source folder for Engineer 1's work
add_executable(core_engine core_engine/src/main.cpp)

# 4. Link OpenCV to the Core Engine
target_link_libraries(core_engine PRIVATE ${OpenCV_LIBS})