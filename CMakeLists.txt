CMAKE_MINIMUM_REQUIRED(VERSION 3.10.1)

project(skb_solver)

#
# C++ standard
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Debug build
#

set(CMAKE_BUILD_TYPE Debug)

#
# Debug flags
#

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

#
# Includes
#

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#
# Cross compilation (optional)
#

if(NOT DEFINED TARGET_OS)
    set(TARGET_OS "native")
endif()

if(TARGET_OS STREQUAL "windows")

    set(CMAKE_SYSTEM_NAME Windows)
    set(CMAKE_SYSTEM_PROCESSOR x86_64)

    set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
    set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

elseif(TARGET_OS STREQUAL "linux")

    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

endif()

#
# Collect all source files
#

file(GLOB_RECURSE MY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

#
# Remove main files to avoid duplicate symbols
#

list(REMOVE_ITEM MY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_benchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_evolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_test_solver.cpp
)

#
# Interactive solver executable
#

add_executable(
    sokoban_solver
    src/main.cpp
    ${MY_SRC}
)

#
# Benchmark executable
#

add_executable(
    solver_benchmark
    src/main_benchmark.cpp
    ${MY_SRC}
)

#
# Evolutionary generator executable
#

add_executable(
    evolution_generator
    src/main_evolution.cpp
    ${MY_SRC}
)

#
# Common compiler flags
#

target_compile_options(
    sokoban_solver
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -g
    -O0
)

target_compile_options(
    solver_benchmark
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -g
    -O0
)

target_compile_options(
    evolution_generator
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -g
    -O0
)

#
# Test solver executable
#

add_executable(
    test_solver
    src/main_test_solver.cpp
    ${MY_SRC}
)

target_compile_options(
    test_solver
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -g
    -O0
)