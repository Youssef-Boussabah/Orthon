cmake_minimum_required(VERSION 3.16)

# This discovers runtime Python modules, not native compilation inputs.
file(GLOB_RECURSE _orthon_python_files LIST_DIRECTORIES FALSE
    RELATIVE "${ORTHON_PYTHON_SOURCE_DIR}"
    "${ORTHON_PYTHON_SOURCE_DIR}/*.py")
foreach(_orthon_file IN LISTS _orthon_python_files)
    if(_orthon_file MATCHES "(^|/)(\\.git|\\.venv|__pycache__|\\.pytest_cache|build|dist|_skbuild|_deps)/")
        continue()
    endif()
    get_filename_component(_orthon_directory "${_orthon_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${ORTHON_PYTHON_STAGE_DIR}/${_orthon_directory}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${ORTHON_PYTHON_SOURCE_DIR}/${_orthon_file}"
            "${ORTHON_PYTHON_STAGE_DIR}/${_orthon_file}"
        RESULT_VARIABLE _orthon_copy_status)
    if(NOT _orthon_copy_status EQUAL 0)
        message(FATAL_ERROR "Could not stage Python module ${_orthon_file}")
    endif()
endforeach()
