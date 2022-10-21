cmake_minimum_required(VERSION 3.20)
project(SCSimulationLib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include(GenerateExportHeader)

find_package(CUDAToolkit REQUIRED)

set (SCSIMLIB_COMPONENT_HEADERS
  BasicCombinatorial.cuh
  LFSR.cuh
  Stanh.cuh
  ParallelCounter.cuh
  Btanh.cuh
  MaxApprox.cuh
)

set (SCSIMLIB_COMPONENT_SOURCES
  BasicCombinatorial.cu
  LFSR.cu
  Stanh.cu
  ParallelCounter.cu
  Btanh.cu
  MaxApprox.cu
)

set (SCSIMLIB_SCHEDULING_HEADERS
  Graph.h
  Scheduler.h
  HostAsapScheduler.h
  AsapScheduler.cuh
)

set (SCSIMLIB_SCHEDULING_SOURCES
  Graph.cpp
  HostAsapScheduler.cpp
  AsapScheduler.cu
)

set(SCSIMLIB_PUBLIC_HEADERS
  circuit_component_defines.cuh
  CircuitComponent.cuh
  CombinatorialComponent.cuh
  cuda_base.cuh
  curand_base.cuh
  SequentialComponent.cuh
  StochasticCircuit.cuh
  StochasticCircuitFactory.cuh
  StochasticNumber.cuh
)
  
set(SCSIMLIB_PRIVATE_HEADERS
  libpopcnt.h
  Range.h
)

set(SCSIMLIB_MAIN_SOURCES
  CircuitComponent.cu
  CombinatorialComponent.cu
  StochasticCircuit.cu
  StochasticCircuitFactory.cu
  StochasticNumber.cu
)
  
add_library(scsim-lib SHARED
  ${SCSIMLIB_MAIN_SOURCES}
  ${SCSIMLIB_COMPONENT_SOURCES}
  ${SCSIMLIB_SCHEDULING_SOURCES}
  ${SCSIMLIB_PUBLIC_HEADERS}
  ${SCSIMLIB_PRIVATE_HEADERS}
  ${SCSIMLIB_COMPONENT_HEADERS}
  ${SCSIMLIB_SCHEDULING_HEADERS}
)

target_compile_features(scsim-lib PUBLIC cxx_std_17)
set_target_properties(scsim-lib PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
set_target_properties(scsim-lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(scsim-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(scsim-lib PROPERTIES OUTPUT_NAME "SCSimulation")

target_link_libraries(scsim-lib CUDA::curand)

GENERATE_EXPORT_HEADER(scsim-lib
  BASE_NAME scsimlib
  EXPORT_MACRO_NAME SCSIMAPI
  EXPORT_FILE_NAME library_export.h
)

file(COPY
  ${SCSIMLIB_PUBLIC_HEADERS}
  ${SCSIMLIB_COMPONENT_HEADERS}
  ${SCSIMLIB_SCHEDULING_HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/library_export.h
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_include_directories(scsim-lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(scsim-lib INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/include")

if (MSVC)
  add_definitions(-DUNICODE -D_UNICODE)
  add_compile_options("/std:c++17")
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath)" ${CMAKE_CURRENT_BINARY_DIR})
endif()

source_group("Header Files" FILES ${SCSIMLIB_PUBLIC_HEADERS} ${SCSIMLIB_PRIVATE_HEADERS})
source_group("Source Files" FILES ${SCSIMLIB_MAIN_SOURCES})
source_group("Header Files/Components" FILES ${SCSIMLIB_COMPONENT_HEADERS})
source_group("Header Files/Scheduling" FILES ${SCSIMLIB_SCHEDULING_HEADERS})
source_group("Source Files/Components" FILES ${SCSIMLIB_COMPONENT_SOURCES})
source_group("Source Files/Scheduling" FILES ${SCSIMLIB_SCHEDULING_SOURCES})

install(TARGETS scsim-lib
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib/SCSimulationLib
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/ TYPE INCLUDE)
