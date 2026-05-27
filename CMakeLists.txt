CMAKE_MINIMUM_REQUIRED(VERSION 3.10.1)

project(skb_solver)

if(NOT DEFINED TARGET_OS)
    set(TARGET_OS "native")
endif()

# 根据目标平台设置交叉编译工具链
if(TARGET_OS STREQUAL "windows")

    # Windows 交叉编译设置
    SET(CMAKE_SYSTEM_NAME Windows)
    SET(CMAKE_SYSTEM_PROCESSOR x86_64)

    SET(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
    SET(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)

elseif(TARGET_OS STREQUAL "linux")

    # Linux ARM 交叉编译设置
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_SYSTEM_PROCESSOR aarch64)

    SET(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    SET(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

else()

    # 本机编译
endif()

SET(CMAKE_BUILD_TYPE "Release")

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#
# Sources comunes
#

FILE(GLOB_RECURSE MY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

#
# Remover los mains para evitar conflicto
#

list(REMOVE_ITEM MY_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_benchmark.cpp
)

#
# Solver interactivo original
#

add_executable(
    sokoban_solver
    src/main.cpp
    ${MY_SRC}
)

#
# Solver benchmark automático
#

add_executable(
    solver_benchmark
    src/main_benchmark.cpp
    ${MY_SRC}
)

#
# Flags
#

target_compile_options(
    sokoban_solver
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -O3
)

target_compile_options(
    solver_benchmark
    PRIVATE
    -Wall
    -Wextra
    -pedantic
    -O3
)

#
# Static linking
#

target_link_options(
    sokoban_solver
    PRIVATE
    -static
    -s
)

target_link_options(
    solver_benchmark
    PRIVATE
    -static
    -s
)