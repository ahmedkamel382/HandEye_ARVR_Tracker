cmake_minimum_required(VERSION 3.15)

# Standalone DSP builds intentionally do not load the repository's AI dependencies.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(HandEyeDSP LANGUAGES CXX)
    include(CTest)
endif()

add_library(dsp_filter STATIC src/one_euro_filter.cpp)
target_include_directories(dsp_filter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(dsp_filter PUBLIC cxx_std_17)
set_target_properties(dsp_filter PROPERTIES CXX_EXTENSIONS OFF)

set(dsp_local_targets dsp_filter)
if(BUILD_TESTING)
    add_executable(dsp_filter_tests tests/test_one_euro_filter.cpp)
    target_link_libraries(dsp_filter_tests PRIVATE dsp_filter)
    add_test(NAME dsp_filter_tests COMMAND dsp_filter_tests)
    list(APPEND dsp_local_targets dsp_filter_tests)

    # These tests exercise the real, dependency-free core adapter and freshness
    # gate. A standalone DSP checkout without the sibling core still builds its
    # mathematical library/tests without Python, OpenCV or Windows dependencies.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../core_engine/include/dsp_adapter.hpp")
        add_executable(dsp_trace_recorder_tests tests/test_trace_recorder.cpp)
        target_link_libraries(dsp_trace_recorder_tests PRIVATE dsp_filter)
        target_include_directories(dsp_trace_recorder_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../core_engine/include")
        target_compile_definitions(dsp_trace_recorder_tests PRIVATE CORE_HAVE_DSP_FILTER=1)
        add_test(NAME dsp_trace_recorder_tests COMMAND dsp_trace_recorder_tests)
        list(APPEND dsp_local_targets dsp_trace_recorder_tests)
    endif()
endif()

option(DSP_BUILD_BENCHMARKS "Build the independent DSP microbenchmark" OFF)
if(DSP_BUILD_BENCHMARKS)
    add_executable(dsp_filter_benchmark tests/benchmark_one_euro_filter.cpp)
    target_link_libraries(dsp_filter_benchmark PRIVATE dsp_filter)
    target_compile_definitions(dsp_filter_benchmark PRIVATE
        DSP_BUILD_CONFIGURATION="$<CONFIG>"
        DSP_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        DSP_ARCHITECTURE="${CMAKE_SYSTEM_PROCESSOR}")
    list(APPEND dsp_local_targets dsp_filter_benchmark)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../core_engine/include/dsp_adapter.hpp")
        add_executable(dsp_trace_recorder_benchmark tests/benchmark_trace_recorder.cpp)
        target_link_libraries(dsp_trace_recorder_benchmark PRIVATE dsp_filter)
        target_include_directories(dsp_trace_recorder_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../core_engine/include")
        target_compile_definitions(dsp_trace_recorder_benchmark PRIVATE
            CORE_HAVE_DSP_FILTER=1
            DSP_BUILD_CONFIGURATION="$<CONFIG>"
            DSP_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
            DSP_ARCHITECTURE="${CMAKE_SYSTEM_PROCESSOR}")
        list(APPEND dsp_local_targets dsp_trace_recorder_benchmark)
    endif()
endif()

# Apply warnings only to Engineer 4 targets, never to teammate targets.
foreach(dsp_target IN LISTS dsp_local_targets)
    set_target_properties(${dsp_target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${dsp_target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${dsp_target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()
