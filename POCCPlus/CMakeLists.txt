cmake_minimum_required(VERSION 3.16)
project(BMSSP LANGUAGES CXX)

# Require C++17 (needed for structured bindings, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings
if (MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Main executable
add_executable(bmssp
    bmssp.cpp
)

# If you want timing/benchmark builds, enable optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
