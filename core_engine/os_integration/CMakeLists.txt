cmake_minimum_required(VERSION 3.15)

# Engineer 5: OS integration & GUI. Builds on its own (like dsp_filter) so it
# never depends on the root project's OpenCV/Pybind11 setup:
#   cmake -S core_engine/os_integration -B out/build/os-debug -G Ninja
# It also works as add_subdirectory(core_engine/os_integration) from the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(HandEyeOsIntegration LANGUAGES CXX)
    include(CTest)
endif()

option(OSI_BUILD_APPS "Build the tray app and os_demo (Windows only)" ON)
option(OSI_BUILD_BENCHMARKS "Build the per-frame latency benchmark" OFF)

# ---------------------------------------------------------------- portable core
add_library(os_integration STATIC
    src/types.cpp
    src/screen_mapper.cpp
    src/gesture_interpreter.cpp
    src/cursor_stabilizer.cpp
    src/settings.cpp
    src/os_controller.cpp
    src/calibration.cpp)
target_include_directories(os_integration PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(os_integration PUBLIC cxx_std_17)

set(osi_local_targets os_integration)

# ---------------------------------------------------------------- Win32 layer
if(WIN32)
    add_library(os_integration_win32 STATIC src/win32_input.cpp)
    target_link_libraries(os_integration_win32 PUBLIC os_integration user32)
    target_compile_definitions(os_integration_win32 PUBLIC
        WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
    list(APPEND osi_local_targets os_integration_win32)
endif()

# ---------------------------------------------------------------- tests
if(BUILD_TESTING)
    add_executable(os_integration_tests tests/test_os_integration.cpp)
    target_link_libraries(os_integration_tests PRIVATE os_integration)
    if(WIN32)
        target_link_libraries(os_integration_tests PRIVATE os_integration_win32)
        target_compile_definitions(os_integration_tests PRIVATE OSI_HAVE_WIN32=1)
    endif()
    add_test(NAME os_integration_tests COMMAND os_integration_tests)
    list(APPEND osi_local_targets os_integration_tests)
endif()

if(OSI_BUILD_BENCHMARKS)
    add_executable(os_integration_benchmark tests/benchmark_os_controller.cpp)
    target_link_libraries(os_integration_benchmark PRIVATE os_integration)
    list(APPEND osi_local_targets os_integration_benchmark)
endif()

# ---------------------------------------------------------------- applications
if(WIN32 AND OSI_BUILD_APPS)
    # Phase 1 deliverable: prove we can drive the Windows cursor.
    add_executable(os_demo tools/os_demo.cpp)
    target_link_libraries(os_demo PRIVATE os_integration_win32 winmm)
    list(APPEND osi_local_targets os_demo)

    # Tray app + calibration wizard. The library lets the core engine host
    # the tray in-process and feed it frames with TrayApp::submit_frame().
    add_library(os_integration_tray STATIC
        app/tray_app.cpp
        app/calibration_wizard.cpp
        app/app_log.cpp)
    target_include_directories(os_integration_tray PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/app")
    target_link_libraries(os_integration_tray PUBLIC os_integration_win32 shell32 gdi32 ole32)

    add_executable(handeye_tray WIN32 app/tray_main.cpp)
    target_link_libraries(handeye_tray PRIVATE os_integration_tray)
    list(APPEND osi_local_targets os_integration_tray)
    list(APPEND osi_local_targets handeye_tray)
endif()

# Warnings apply to Engineer 5 targets only, never to teammates' targets.
foreach(osi_target IN LISTS osi_local_targets)
    set_target_properties(${osi_target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${osi_target} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${osi_target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()
