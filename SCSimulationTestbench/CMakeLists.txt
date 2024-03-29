cmake_minimum_required(VERSION 3.20)
project(SCSimulationTestbench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(SCSIMTB_HEADERS
  BitwiseAbleComponents.cuh
  ChainedInverterTestbench.cuh
  CycleTestbench.cuh
  InverterTestbench.cuh
  MLPLayerCountTestbench.cuh
  MLPLayerSizeTestbench.cuh
  MuxNTestbench.cuh
  SquarerTestbench.cuh
  StanhTestbench.cuh
  Testbench.cuh
  FCLayer.cuh
  ConvolutionLayer.cuh
  MaxPoolLayer.cuh
  SoftwareFCLayer.h
  NumericCSVReader.h
  BinaryDoubleReader.h
  PCBtanhTestbench.cuh
  CNNTestbench.cuh
)

set(SCSIMTB_SOURCES
  main.cu
  NumericCSVReader.cpp
  BinaryDoubleReader.cpp
)
  
add_executable(scsim-testbench ${SCSIMTB_SOURCES} ${SCSIMTB_HEADERS})

target_compile_features(scsim-testbench PUBLIC cxx_std_17)
set_target_properties(scsim-testbench PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
set_target_properties(scsim-testbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(scsim-testbench PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(scsim-testbench PROPERTIES OUTPUT_NAME "SCSimulationTestbench")

target_link_libraries(scsim-testbench scsim-lib)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(scsim-testbench OpenMP::OpenMP_CXX)
endif()

if (MSVC)
  add_definitions(-DUNICODE -D_UNICODE)
  get_target_property(scsimlib_include scsim-lib INTERFACE_INCLUDE_DIRECTORIES)
  set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$(VC_IncludePath);$(WindowsSDK_IncludePath)" ${scsimlib_include})
endif()

source_group("Header Files" FILES ${SCSIMTB_HEADERS})
install(TARGETS scsim-testbench DESTINATION bin)
