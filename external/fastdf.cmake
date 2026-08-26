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

    set(FASTDF_MEMORY_PATCH "${CMAKE_CURRENT_LIST_DIR}/fastdf-memory-sink.patch")
    execute_process(
        COMMAND git apply --check "${FASTDF_MEMORY_PATCH}"
        WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
        RESULT_VARIABLE FASTDF_PATCH_CHECK
        OUTPUT_QUIET
        ERROR_QUIET)
    if(FASTDF_PATCH_CHECK EQUAL 0)
        execute_process(
            COMMAND git apply "${FASTDF_MEMORY_PATCH}"
            WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
            RESULT_VARIABLE FASTDF_PATCH_RESULT)
        if(NOT FASTDF_PATCH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply the FastDF in-memory particle sink patch")
        endif()
    else()
        execute_process(
            COMMAND git apply --reverse --check "${FASTDF_MEMORY_PATCH}"
            WORKING_DIRECTORY "${fastdf_SOURCE_DIR}"
            RESULT_VARIABLE FASTDF_PATCH_REVERSE_CHECK
            OUTPUT_QUIET
            ERROR_QUIET)
        if(NOT FASTDF_PATCH_REVERSE_CHECK EQUAL 0)
            message(FATAL_ERROR "FastDF source does not match the pinned in-memory sink patch")
        endif()
    endif()

    # FastDF uses a fixed HDF5 chunk length of 65536.  HDF5 rejects that
    # layout when generating fewer particles (including the required 32^3
    # smoke case), while FastDF does not check the H5Dcreate return value.
    # Cap the chunk to the actual per-file particle count.
    set(FASTDF_RUNNER "${fastdf_SOURCE_DIR}/src/runner.c")
    file(READ "${FASTDF_RUNNER}" FASTDF_RUNNER_CONTENTS)
    set(FASTDF_RUNNER_ORIGINAL "${FASTDF_RUNNER_CONTENTS}")
    string(REPLACE
        "const hsize_t vchunk[2] = {HDF5_CHUNK_SIZE, 3};"
        "const hsize_t vchunk[2] = {parts_in_file < HDF5_CHUNK_SIZE ? parts_in_file : HDF5_CHUNK_SIZE, 3};"
        FASTDF_RUNNER_CONTENTS "${FASTDF_RUNNER_CONTENTS}")
    string(REPLACE
        "const hsize_t schunk[1] = {HDF5_CHUNK_SIZE};"
        "const hsize_t schunk[1] = {parts_in_file < HDF5_CHUNK_SIZE ? parts_in_file : HDF5_CHUNK_SIZE};"
        FASTDF_RUNNER_CONTENTS "${FASTDF_RUNNER_CONTENTS}")
    if(NOT FASTDF_RUNNER_CONTENTS MATCHES "H5Pset_obj_track_times\\(h_prop_vec")
        string(REPLACE
            "hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);"
            "hid_t h_prop_vec = H5Pcreate(H5P_DATASET_CREATE);\n        H5Pset_obj_track_times(h_prop_vec, 0);"
            FASTDF_RUNNER_CONTENTS "${FASTDF_RUNNER_CONTENTS}")
        string(REPLACE
            "hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);"
            "hid_t h_prop_sca = H5Pcreate(H5P_DATASET_CREATE);\n        H5Pset_obj_track_times(h_prop_sca, 0);"
            FASTDF_RUNNER_CONTENTS "${FASTDF_RUNNER_CONTENTS}")
        string(REPLACE
            "h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);"
            "hid_t h_prop_grp = H5Pcreate(H5P_GROUP_CREATE);\n        H5Pset_obj_track_times(h_prop_grp, 0);\n        h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, h_prop_grp, H5P_DEFAULT);\n        H5Pclose(h_prop_grp);"
            FASTDF_RUNNER_CONTENTS "${FASTDF_RUNNER_CONTENTS}")
    endif()
    if(NOT FASTDF_RUNNER_CONTENTS STREQUAL FASTDF_RUNNER_ORIGINAL)
        file(WRITE "${FASTDF_RUNNER}" "${FASTDF_RUNNER_CONTENTS}")
    endif()

    add_subdirectory(${fastdf_SOURCE_DIR} ${fastdf_BINARY_DIR})
    set(WITH_CLASS 0)
endif()
