cmake_minimum_required(VERSION 3.10.1)

project(skb_solver)

#
# C++ standard
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Release build
#

set(CMAKE_BUILD_TYPE Release)

#
# Release flags
#

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_irace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_batch.cpp     # <- MODIFICADO: Evita conflicto en los demas binarios
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
# Test solver executable
#

add_executable(
    test_solver
    src/main_test_solver.cpp
    ${MY_SRC}
)

#
# IRace generator executable
#

add_executable(
    irace_generator
    src/main_irace.cpp
    ${MY_SRC}
)

#
# Batch solver executable (Procesamiento por lotes .sok)
#

add_executable(
    batch_solver
    src/main_batch.cpp                                  # <- MODIFICADO: Añadido nuevo binario batch
    ${MY_SRC}
)

#
# Common compiler flags
#

foreach(target
    sokoban_solver
    solver_benchmark
    evolution_generator
    test_solver
    irace_generator
    batch_solver                                        # <- MODIFICADO: Añadido al bucle de optimizacion
)
    target_compile_options(
        ${target}
        PRIVATE
        -Wall
        -Wextra
        -pedantic
        -O3
        -march=native
        -flto
        -DNDEBUG
    )

    target_link_options(
        ${target}
        PRIVATE
        -flto
    )
endforeach()