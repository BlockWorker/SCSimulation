cmake_minimum_required(VERSION 3.22)
project(SCSimulation)

set(CMAKE_CUDA_ARCHITECTURES 52)

add_subdirectory(SCSimulationLib)
add_subdirectory(SCSimulationTestbench)
