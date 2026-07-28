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
# LibTorch (PyTorch C++) Integration
#
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_experiment1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_shell.cpp     # Nuevo main para pruebas
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_es_dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_playable_dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_experiment.cpp
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
# Experiment runner executable (salida: fitness;hash)
#

add_executable(
    experiment_runner
    src/main_experiment.cpp
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
# Shell test executable
#

add_executable(
    test_shell
    src/main_shell.cpp
    ${MY_SRC}
)

add_executable(
    dataset_generator
    src/main_dataset.cpp
    ${MY_SRC}
)

add_executable(
    playable_dataset_generator
    src/main_playable_dataset.cpp
    ${MY_SRC}
)

add_executable(
    es_dataset_generator
    src/main_es_dataset.cpp
    ${MY_SRC}
)

#
# Common compiler flags
#

add_executable(
    test_tensor_parity
    tests/test_tensor_parity.cpp
    ${MY_SRC}
)


foreach(target
    sokoban_solver
    solver_benchmark
    evolution_generator
    test_solver
    irace_generator
    batch_solver                                        # <- MODIFICADO: Añadido al bucle de optimizacion
    test_shell
    dataset_generator
    playable_dataset_generator
    es_dataset_generator
    experiment_runner
    test_tensor_parity
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
        -pthread
    )

    target_link_options(
        ${target}
        PRIVATE
        -flto
        -pthread
    )

    target_link_libraries(
        ${target}
        PRIVATE
        ${TORCH_LIBRARIES}
    )
endforeach()