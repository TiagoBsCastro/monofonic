cmake_minimum_required(VERSION 3.11)
include(FetchContent)

FetchContent_Declare(
    fastdf
    GIT_REPOSITORY https://github.com/wullm/fastdf.git
    GIT_TAG 2da480cf69076fa67801ccc834113b4111af7d6b
    GIT_SHALLOW YES
    GIT_PROGRESS TRUE
)

FetchContent_GetProperties(fastdf)
if(NOT fastdf_POPULATED)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Populate(fastdf)

    # Keep all monofonIC-specific FastDF changes atomic. The reverse check
    # makes repeated CMake configuration idempotent while still detecting a
    # dependency tree that no longer matches the pinned revision.
    set(FASTDF_INTEGRATION_PATCH
        "${CMAKE_CURRENT_LIST_DIR}/fastdf-monofonic-integration.patch")
    execute_process(
        COMMAND git apply --check "${FASTDF_INTEGRATION_PATCH}"
        WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
        RESULT_VARIABLE FASTDF_PATCH_CHECK
        OUTPUT_QUIET
        ERROR_QUIET)
    if(FASTDF_PATCH_CHECK EQUAL 0)
        execute_process(
            COMMAND git apply "${FASTDF_INTEGRATION_PATCH}"
            WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
            RESULT_VARIABLE FASTDF_PATCH_RESULT)
        if(NOT FASTDF_PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply the FastDF monofonIC integration patch")
        endif()
    else()
        execute_process(
            COMMAND git apply --reverse --check
                    "${FASTDF_INTEGRATION_PATCH}"
            WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
            RESULT_VARIABLE FASTDF_PATCH_REVERSE_CHECK
            OUTPUT_QUIET
            ERROR_QUIET)
        if(NOT FASTDF_PATCH_REVERSE_CHECK EQUAL 0)
            message(FATAL_ERROR
                "FastDF source does not match the pinned monofonIC integration patch")
        endif()
    endif()

    add_subdirectory("${fastdf_SOURCE_DIR}" "${fastdf_BINARY_DIR}")
    set(WITH_CLASS 0)
endif()
