cmake_minimum_required(VERSION 3.20)
project(compress3d C)

set(CMAKE_C_STANDARD 11)

# The core C3D library — no external dependencies
add_library(compress3d STATIC src/compress3d.c src/c3d_net.c)
target_include_directories(compress3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link pthreads (required for batch/shard parallel APIs)
find_package(Threads REQUIRED)
target_link_libraries(compress3d PUBLIC Threads::Threads m)

# Enable SIMD flags if available
include(CheckCCompilerFlag)
check_c_compiler_flag(-mavx2 HAS_AVX2)
if(HAS_AVX2)
    target_compile_options(compress3d PRIVATE -mavx2 -mfma)
endif()

# Tests
option(C3D_BUILD_TESTS "Build compress3d tests" ON)
if(C3D_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
