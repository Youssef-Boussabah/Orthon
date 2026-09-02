cmake_minimum_required(VERSION 3.16)
# VERSION is release metadata and must match `version` in pyproject.toml, which
# is the number the built distribution carries. Two literals that can drift are
# two chances to ship a wrong one, so
# tests/python/test_build_architecture.py compares them.
project(Orthon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Library sources ------------------------------------------------------
#
# Listed explicitly, never globbed. A glob hides two things that matter here:
# adding or deleting a file does not re-run configure, so a stale build tree
# silently compiles the wrong set; and a dead source stays in the library
# because nothing ever names it. The lists below are the build's declaration of
# what the library IS, and tests/python/test_build_architecture.py checks them
# against the tree on disk.

set(ORTHON_CORE_SOURCES
    src/autograd/grad_mode.cpp
    src/autograd/grad_utils.cpp
    src/autograd/node.cpp
    src/core/dtype.cpp
    src/core/float_formats.cpp
    src/core/memory_pool.cpp
    src/core/shape.cpp
    src/core/storage.cpp
    src/core/strides.cpp
    src/core/tensor.cpp
    src/core/tensor_impl.cpp
    src/device/device.cpp
    src/device/device_guard.cpp
    src/memory/allocator.cpp
    src/memory/cpu_allocator.cpp
    src/ops/cpu/binary_ops.cpp
    src/ops/cpu/cast_ops.cpp
    src/ops/cpu/conv2d_ops.cpp
    src/ops/cpu/cross_entropy_ops.cpp
    src/ops/cpu/embedding_ops.cpp
    src/ops/cpu/fill_ops.cpp
    src/ops/cpu/fusion_ops.cpp
    src/ops/cpu/index_ops.cpp
    src/ops/cpu/matmul_ops.cpp
    src/ops/cpu/maxpool2d_ops.cpp
    src/ops/cpu/nll_loss_ops.cpp
    src/ops/cpu/reduce_ops.cpp
    src/ops/cpu/scalar_ops.cpp
    src/ops/cpu/shape_ops.cpp
    src/ops/cpu/slice_ops.cpp
    src/ops/cpu/softmax_ops.cpp
    src/ops/cpu/unary_ops.cpp
    src/utils/logging.cpp
)

set(ORTHON_CUDA_SOURCES
    src/ops/cuda/binary_kernels.cu
    src/ops/cuda/cast_kernels.cu
    src/ops/cuda/conv2d_cuda.cu
    src/ops/cuda/cross_entropy_kernels.cu
    src/ops/cuda/embedding_kernels.cu
    src/ops/cuda/fusion_kernels.cu
    src/ops/cuda/matmul_kernels.cu
    src/ops/cuda/maxpool2d_kernels.cu
    src/ops/cuda/nll_loss_kernels.cu
    src/ops/cuda/reduce_kernels.cu
    src/ops/cuda/softmax_kernels.cu
    src/ops/cuda/unary_kernels.cu
)

foreach(_src IN LISTS ORTHON_CORE_SOURCES ORTHON_CUDA_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        message(FATAL_ERROR "Listed source ${_src} does not exist.")
    endif()
endforeach()

add_library(orthon ${ORTHON_CORE_SOURCES})
add_library(Orthon::orthon ALIAS orthon)
target_include_directories(orthon PUBLIC include)

# This static library is linked into the _C Python MODULE. On Linux that link
# requires every object to be position-independent: without -fPIC the
# thread_local grad-mode state compiles to a local-exec TLS relocation
# (R_X86_64_TPOFF32) that the linker rejects in a shared object. Set the
# property on the target itself rather than expecting builders to pass
# CMAKE_POSITION_INDEPENDENT_CODE; on MSVC the property is a no-op.
set_target_properties(orthon PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Optional CPU SIMD kernels -------------------------------------------
option(WITH_AVX2 "Enable AVX2 SIMD on CPU" OFF)
message(STATUS "WITH_AVX2 = ${WITH_AVX2}")
if(WITH_AVX2)
    if(MSVC)
        target_compile_options(orthon PRIVATE /arch:AVX2)
    else()
        target_compile_options(orthon PRIVATE -mavx2)
    endif()
endif()

# ---- Optional CUDA kernels -----------------------------------------------
set(WITH_CUDA "AUTO" CACHE STRING "Build CUDA kernels: AUTO, ON, or OFF")
set_property(CACHE WITH_CUDA PROPERTY STRINGS AUTO ON OFF)
set(WITH_NCCL "AUTO" CACHE STRING "Build NCCL distributed collectives: AUTO, ON, or OFF")
set_property(CACHE WITH_NCCL PROPERTY STRINGS AUTO ON OFF)

string(TOUPPER "${WITH_CUDA}" _ORTHON_WITH_CUDA)
string(TOUPPER "${WITH_NCCL}" _ORTHON_WITH_NCCL)
set(_ORTHON_BOOL_MODES AUTO ON OFF)
if(NOT "${_ORTHON_WITH_CUDA}" IN_LIST _ORTHON_BOOL_MODES)
    message(FATAL_ERROR "WITH_CUDA must be AUTO, ON, or OFF (got '${WITH_CUDA}')")
endif()
if(NOT "${_ORTHON_WITH_NCCL}" IN_LIST _ORTHON_BOOL_MODES)
    message(FATAL_ERROR "WITH_NCCL must be AUTO, ON, or OFF (got '${WITH_NCCL}')")
endif()

set(_ORTHON_CUDA_ENABLED OFF)
if(_ORTHON_WITH_CUDA STREQUAL "ON")
    set(_ORTHON_CUDA_ENABLED ON)
elseif(_ORTHON_WITH_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(_ORTHON_CUDA_ENABLED ON)
            message(STATUS "CUDA auto-detect: enabled (${CMAKE_CUDA_COMPILER})")
        else()
            message(STATUS "CUDA auto-detect: nvcc found but CUDAToolkit package was not found; building CPU-only")
        endif()
    else()
        message(STATUS "CUDA auto-detect: CUDA compiler not found; building CPU-only")
    endif()
else()
    message(STATUS "CUDA disabled by WITH_CUDA=OFF")
endif()

set(_ORTHON_NCCL_ENABLED OFF)

message(STATUS "WITH_CUDA = ${WITH_CUDA}")
message(STATUS "WITH_NCCL = ${WITH_NCCL}")
if(_ORTHON_CUDA_ENABLED)
    # CUDA architecture selection.
    #
    # CMP0104 (CMake 3.18+) is what makes CMake initialise
    # CMAKE_CUDA_ARCHITECTURES from the compiler. Left unset, a CUDA configure
    # warns that CUDA_ARCHITECTURES is empty and the build silently falls back
    # to whatever nvcc happens to default to - a default that changes between
    # toolkit releases. Opting in makes the target architecture an explicit,
    # visible value (printed below) instead of an accident.
    #
    # Deliberately NOT hard-coded to one SM: which architecture a build should
    # target belongs to whoever is building, not to this file. Override on the
    # command line or in a preset, e.g.
    #     -DCMAKE_CUDA_ARCHITECTURES=native      (CMake 3.24+, this machine's GPUs)
    #     -DCMAKE_CUDA_ARCHITECTURES="80;86;89"  (a portable fat binary)
    # The 3.16 floor still configures: the policy simply does not exist there,
    # and the pre-3.18 behaviour is unchanged.
    if(POLICY CMP0104)
        cmake_policy(SET CMP0104 NEW)
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    message(STATUS "CUDA compiler : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # CUDAToolkit provides include dirs + CUDA::cudart for host-side
    # runtime API calls (cudaMemcpy, etc.) from plain .cpp files.
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit include: ${CUDAToolkit_INCLUDE_DIRS}")

    list(LENGTH ORTHON_CUDA_SOURCES _orthon_cuda_source_count)
    message(STATUS "CUDA sources listed: ${_orthon_cuda_source_count}")
    target_sources(orthon PRIVATE ${ORTHON_CUDA_SOURCES})

    # `orthon` is created above, before the CUDA language exists, so its
    # CUDA_ARCHITECTURES property was initialised from a variable that had no
    # value yet. Propagate the resolved value now - otherwise the property
    # stays empty and nvcc is invoked with no -gencode at all.
    if(CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(orthon PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Expose CUDA include dirs so cl.exe can find cuda_runtime_api.h
    # in .cpp files compiled with WITH_CUDA.
    target_include_directories(orthon PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # Link cudart for runtime API calls and cuBLAS for optimized matmul paths.
    target_link_libraries(orthon PUBLIC CUDA::cudart CUDA::cublas)

    # Propagate WITH_CUDA to all targets that link orthon.
    # PUBLIC ensures _C (which links orthon) also sees the definition.
    target_compile_definitions(orthon PUBLIC WITH_CUDA)

    if(_ORTHON_WITH_NCCL STREQUAL "OFF")
        message(STATUS "NCCL support disabled by WITH_NCCL=OFF")
    else()
        find_path(NCCL_INCLUDE_DIR nccl.h)
        find_library(NCCL_LIBRARY nccl)
        if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            set(_ORTHON_NCCL_ENABLED ON)
        endif()
    endif()

    if(_ORTHON_WITH_NCCL STREQUAL "ON" AND NOT _ORTHON_NCCL_ENABLED)
        message(FATAL_ERROR "WITH_NCCL=ON requires NCCL headers and library")
    endif()

    if(_ORTHON_NCCL_ENABLED)
        message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
        message(STATUS "NCCL library: ${NCCL_LIBRARY}")
        target_include_directories(orthon PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(orthon PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(orthon PUBLIC WITH_NCCL)
        message(STATUS "NCCL support enabled")
    elseif(_ORTHON_WITH_NCCL STREQUAL "AUTO")
        message(STATUS "NCCL auto-detect: headers/library not found; building without NCCL")
    endif()
    message(STATUS "CUDA support enabled")
else()
    if(_ORTHON_WITH_NCCL STREQUAL "ON")
        message(FATAL_ERROR "WITH_NCCL=ON requires WITH_CUDA=ON")
    endif()
    if(_ORTHON_WITH_NCCL STREQUAL "AUTO")
        message(STATUS "NCCL auto-detect: skipped because CUDA is disabled")
    endif()
    message(STATUS "CUDA support disabled")
endif()

# ---- Native GoogleTest C++ suite ------------------------------------------
#
# The contract here is: if testing is on, tests EXIST, or configuration fails.
# `ctest` reports the same exit code for "zero tests registered" as for a
# passing run, so a silently empty suite is indistinguishable from a green one
# in automation. Every path below therefore either produces test targets or
# stops the configure.

# scikit-build-core defines SKBUILD when it drives CMake to produce the Python
# wheel. Installing the Python package must never pull in a C++ test framework,
# so testing is forced off here BEFORE include(CTest) can default it to ON.
if(SKBUILD)
    set(BUILD_TESTING OFF CACHE BOOL "Disabled for scikit-build-core wheel builds" FORCE)
    message(STATUS "SKBUILD detected: native C++ tests and GoogleTest disabled")
endif()

# Standard CMake testing model: defines the BUILD_TESTING option (default ON)
# and calls enable_testing(). Deliberately not a bespoke project-specific
# switch. The Python/runtime presets pin BUILD_TESTING=OFF; the cpp-* presets
# pin it ON.
include(CTest)

# Escape hatch for offline / vendored-dependency environments. With this OFF
# and no compatible system GoogleTest, configuration FAILS rather than silently
# registering nothing.
option(ORTHON_FETCH_GTEST
       "Download a pinned GoogleTest when no compatible one is installed" ON)

# Pinned to an immutable COMMIT, not a tag. A tag is a movable reference: an
# upstream retag would silently change what this build downloads, and nothing
# here would notice. The commit below is what refs/tags/v1.18.0 resolves to,
# verified against https://github.com/google/googletest. v1.18.0 is a stable
# release; it requires C++17 (this project already sets CMAKE_CXX_STANDARD 17)
# and declares cmake_minimum_required 3.16/3.13, both satisfied here.
set(ORTHON_GTEST_VERSION "v1.18.0" CACHE STRING
    "Human-readable GoogleTest release matching ORTHON_GTEST_COMMIT")
set(ORTHON_GTEST_COMMIT "063de7e9578f82b369302001269680b4b1553359" CACHE STRING
    "Immutable GoogleTest commit fetched when ORTHON_FETCH_GTEST is ON")

if(BUILD_TESTING)
    # Prefer an already-installed GoogleTest. CONFIG mode only: the module-mode
    # fallback happily reports success for an MSYS2/MinGW GTest that cannot
    # link against an MSVC build.
    find_package(GTest CONFIG QUIET)

    if(TARGET GTest::gtest_main)
        message(STATUS "GoogleTest: using installed package (${GTest_DIR})")
    elseif(ORTHON_FETCH_GTEST)
        message(STATUS "GoogleTest: no installed package found; fetching pinned "
                       "${ORTHON_GTEST_VERSION} (${ORTHON_GTEST_COMMIT})")
        include(FetchContent)

        # MSVC runtime library. VERIFIED, not assumed: this project compiles
        # with -MD (MultiThreadedDLL - confirmed in the generated build rules),
        # while googletest/cmake/internal_utils.cmake forces
        # CMAKE_MSVC_RUNTIME_LIBRARY to "MultiThreaded" (/MT) for a static
        # build unless gtest_force_shared_crt is ON. Leaving it OFF produces
        # LNK2038 RuntimeLibrary mismatches at test-link time.
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        # Never contribute GoogleTest to this project's install() set.
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

        # GIT_SHALLOW is deliberately OFF: a shallow fetch of a bare commit
        # only works where the server allows fetching an arbitrary SHA-1, so
        # combining it with a commit pin would trade reproducibility for a
        # host-dependent failure. Correctness of the pin wins.
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        ${ORTHON_GTEST_COMMIT}
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR
            "BUILD_TESTING=ON but no compatible GoogleTest was found and "
            "ORTHON_FETCH_GTEST=OFF.\n"
            "Resolve one of:\n"
            "  * install GoogleTest and point CMAKE_PREFIX_PATH/GTest_DIR at it,\n"
            "  * re-run with -DORTHON_FETCH_GTEST=ON to download the pinned "
            "${ORTHON_GTEST_VERSION},\n"
            "  * re-run with -DBUILD_TESTING=OFF to build without the native "
            "test suite.\n"
            "Configuration fails here on purpose: a build that silently "
            "registers zero C++ tests is worse than one that stops.")
    endif()

    # Both resolution paths must end at the same imported/alias target. If they
    # do not, stop rather than emit test targets that will fail to link.
    if(NOT TARGET GTest::gtest_main)
        message(FATAL_ERROR
            "GoogleTest was resolved but target GTest::gtest_main does not "
            "exist; refusing to register native tests against a broken "
            "dependency.")
    endif()

    include(GoogleTest)

    # gtest_discover_tests registers every TEST() as its own CTest case instead
    # of one opaque per-executable entry. PRE_TEST enumerates at `ctest` time
    # rather than at build time, which keeps a build machine from having to run
    # the CUDA-linked binaries; it needs CMake 3.18+, so it is applied only
    # where available and the 3.16 floor still configures.
    set(_orthon_discovery "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set(_orthon_discovery DISCOVERY_MODE PRE_TEST)
    endif()

    function(orthon_add_cpp_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE orthon GTest::gtest GTest::gtest_main)
        gtest_discover_tests(${target}
            ${_orthon_discovery}
            DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 300
        )
    endfunction()

    # Every source stays its own executable. Merging them would shrink the
    # target list at the cost of losing per-area isolation, so it is not done.
    set(ORTHON_CPP_TESTS
        test_tensor
        test_ops_binary
        test_ops_shape
        test_ops_reduce
        test_ops_unary
        test_ops_to
        test_ops_matmul
        test_indexing
        test_device_guard
        test_ops_scalar
        test_float_formats
        test_identity
    )

    foreach(_t IN LISTS ORTHON_CPP_TESTS)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/${_t}.cpp")
            message(FATAL_ERROR
                "Native test source tests/cpp/${_t}.cpp is missing; refusing "
                "to configure a partial C++ suite.")
        endif()
        orthon_add_cpp_test(${_t} tests/cpp/${_t}.cpp)
    endforeach()

    # BUILD_TESTING=ON plus a successful configure means native tests exist.
    # This catches the list itself being emptied.
    list(LENGTH ORTHON_CPP_TESTS _orthon_test_count)
    if(_orthon_test_count EQUAL 0)
        message(FATAL_ERROR
            "BUILD_TESTING=ON but no native test targets were registered.")
    endif()
    message(STATUS "Native C++ test executables registered: ${_orthon_test_count}")
else()
    message(STATUS "BUILD_TESTING=OFF: native C++ tests and GoogleTest not required")
endif()

# ---- PyBind11 Python bindings ----
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    # Locate pybind11 via the pip-installed copy, then fall back to system.
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_cmake_dir}")
    else()
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(_C
        bindings/module.cpp
        bindings/tensor_bindings.cpp
        bindings/ops_bindings.cpp
        bindings/dtype_bindings.cpp
        bindings/device_bindings.cpp
        bindings/nccl_bindings.cpp
    )
    target_link_libraries(_C PRIVATE orthon)

    if(SKBUILD)
        # scikit-build-core uses install() to place the extension at orthon/_C.*.so.
        install(TARGETS _C DESTINATION orthon)
    else()
        # Plain cmake workflow: place the extension in build/orthon/ for
        # PYTHONPATH=build use. $<1:...> is a generator expression that
        # evaluates to its argument unchanged, preventing multi-config
        # generators (MSVC) from appending a config subdirectory.
        set_target_properties(_C PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/orthon>"
            RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/orthon>"
        )
        configure_file(
            "${CMAKE_SOURCE_DIR}/python/orthon/__init__.py"
            "${CMAKE_BINARY_DIR}/orthon/__init__.py"
            COPYONLY
        )
    endif()
endif()
