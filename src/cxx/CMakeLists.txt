cmake_minimum_required(VERSION 3.10)

project(frank)

set(CMAKE_CXX_STANDARD 17)

add_library(stopwatch external/Stopwatch.h external/Stopwatch.cpp)
add_executable(gridding gridding.cpp)
add_executable(degridding degridding.cpp)

target_link_libraries(gridding idg-util idg-common idg-cuda casa_ms casa_tables casa_casa stopwatch)
