cmake_minimum_required(VERSION 3.20)

project(hand_index_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
endif()

find_package(Python3 3.11 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import site; print(next(p for p in site.getsitepackages() if p.replace(chr(92), '/').endswith('/site-packages')))"
    OUTPUT_VARIABLE HAND_INDEX_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.base_prefix)"
    OUTPUT_VARIABLE HAND_INDEX_PYTHON_BASE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

file(TO_CMAKE_PATH "${HAND_INDEX_SITE_PACKAGES}" HAND_INDEX_SITE_PACKAGES)
file(TO_CMAKE_PATH "${HAND_INDEX_PYTHON_BASE}" HAND_INDEX_PYTHON_BASE)

add_library(hand_index_tracker SHARED
    src/hand_index_tracker.cpp
)

target_include_directories(hand_index_tracker
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(hand_index_tracker PRIVATE HAND_INDEX_TRACKER_BUILD)
target_link_libraries(hand_index_tracker PRIVATE Python3::Python)

set_target_properties(hand_index_tracker PROPERTIES
    OUTPUT_NAME "hand_index_tracker"
    PREFIX ""
    WINDOWS_EXPORT_ALL_SYMBOLS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(MSVC)
    target_compile_options(hand_index_tracker PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(hand_index_tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(HAND_INDEX_BUILD_TESTS "Build the C++ DLL interface smoke test" ON)
if(HAND_INDEX_BUILD_TESTS)
    add_executable(hand_index_smoke tests/hand_index_smoke.cpp)
    target_link_libraries(hand_index_smoke PRIVATE hand_index_tracker)
    target_compile_definitions(hand_index_smoke PRIVATE
        HAND_INDEX_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        HAND_INDEX_TEST_REAL_PYTHON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ai_pipeline/hand_kinematics"
        HAND_INDEX_TEST_REAL_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../assets/hand_landmarker.task"
        HAND_INDEX_TEST_SITE_PACKAGES="${HAND_INDEX_SITE_PACKAGES}"
        HAND_INDEX_TEST_PYTHON_BASE="${HAND_INDEX_PYTHON_BASE}"
    )
    set_target_properties(hand_index_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(MSVC)
        target_compile_options(hand_index_smoke PRIVATE /W4 /permissive- /utf-8)
    endif()

    add_executable(hand_index_real_backend_integration
        tests/real_backend_integration.cpp
    )
    target_link_libraries(hand_index_real_backend_integration PRIVATE
        hand_index_tracker
        windowscodecs
        ole32
    )
    target_compile_definitions(hand_index_real_backend_integration PRIVATE
        HAND_INDEX_TEST_REAL_PYTHON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../ai_pipeline/hand_kinematics"
        HAND_INDEX_TEST_REAL_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../assets/hand_landmarker.task"
        HAND_INDEX_TEST_SITE_PACKAGES="${HAND_INDEX_SITE_PACKAGES}"
    )
    set_target_properties(hand_index_real_backend_integration PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(MSVC)
        target_compile_options(hand_index_real_backend_integration PRIVATE /W4 /permissive- /utf-8)
    endif()
endif()

option(HAND_INDEX_BUILD_EXAMPLE "Build the minimal C++ integration example" ON)
if(HAND_INDEX_BUILD_EXAMPLE)
    add_executable(hand_index_example examples/index_tracker_example.cpp)
    target_link_libraries(hand_index_example PRIVATE hand_index_tracker)
    set_target_properties(hand_index_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    if(MSVC)
        target_compile_options(hand_index_example PRIVATE /W4 /permissive- /utf-8)
    endif()
endif()
