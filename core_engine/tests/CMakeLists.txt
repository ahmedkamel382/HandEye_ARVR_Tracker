# Tests for the core engine. Run them with:   ctest --output-on-failure   (from the build folder)
#   core_unit_tests   always: filter, gaze maths, hand maths (header-only code, no camera, no Python)
#   gaze_hand_parity  needs both compiled forests: the C++ ports versus the REAL gaze_intent.py / hand_tracker.py
#   core_e2e          needs the compiled gaze forest: whole engine on a synthetic video with stand-in extractors
add_executable(core_unit_tests unit/test_all.cpp)
target_include_directories(core_unit_tests PRIVATE "${CMAKE_SOURCE_DIR}/core_engine/include")
if(MSVC)
    target_compile_options(core_unit_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(core_unit_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME core_unit_tests COMMAND core_unit_tests)

# parity: the C++ gaze + hand ports versus the REAL Python files (needs BOTH compiled forests)
if(CORE_HAVE_TREELITE_FOREST AND CORE_HAVE_HAND_FOREST)
    add_executable(core_parity_check parity/parity_check.cpp)
    target_include_directories(core_parity_check PRIVATE "${CMAKE_SOURCE_DIR}/core_engine/include")
    target_link_libraries(core_parity_check PRIVATE gaze_forest hand_forest)
    target_compile_definitions(core_parity_check PRIVATE CORE_HAVE_TREELITE_FOREST=1 CORE_HAVE_HAND_FOREST=1)
    add_test(NAME gaze_hand_parity
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/parity/run_parity.py"
                     --exe "$<TARGET_FILE:core_parity_check>" --ai-root "${CMAKE_SOURCE_DIR}/ai_pipeline")
else()
    message(WARNING "gaze_hand_parity test skipped: it needs both compiled forests (run core_engine/tools/export_forests.py)")
endif()

# end-to-end: whole engine on a synthetic video (gaze forest compiled in; the hand model is a stand-in .pkl)
if(CORE_HAVE_TREELITE_FOREST)
    add_test(NAME core_e2e
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/e2e/run_e2e.py"
                     --exe "$<TARGET_FILE:core_engine>" --ai-root "${CMAKE_CURRENT_SOURCE_DIR}/fake_ai")
endif()

# The adapter around Eng 4's filter (needs the dsp_filter target)
if(TARGET dsp_filter)
    add_executable(core_dsp_adapter_tests unit/test_dsp_adapter.cpp)
    target_include_directories(core_dsp_adapter_tests PRIVATE "${CMAKE_SOURCE_DIR}/core_engine/include")
    target_link_libraries(core_dsp_adapter_tests PRIVATE dsp_filter)
    target_compile_definitions(core_dsp_adapter_tests PRIVATE CORE_HAVE_DSP_FILTER=1)
    if(MSVC)
        target_compile_options(core_dsp_adapter_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(core_dsp_adapter_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME core_dsp_adapter_tests COMMAND core_dsp_adapter_tests)
endif()
