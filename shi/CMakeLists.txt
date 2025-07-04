cmake_minimum_required (VERSION 3.18)
project ("sunshine_hours")

set(CMAKE_CXX_STANDARD 17)

# Set CMake policy to suppress FindCUDA deprecation warning
cmake_policy(SET CMP0146 NEW)

message("Compile with CUDA")
set(CMAKE_CUDA_COMPILER "/home/ftx/miniconda3/envs/qidong/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 80)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -Xcompiler=-fopenmp")

add_executable(sunshine_hours
    src/main.cpp
    src/timer.h
    src/raster.h
    src/raster.cpp
    src/sunshine_hours.cpp
    src/sunshine_hours.h
    src/cuda_core.cuh
    src/cuda_core.cu
)

target_include_directories(sunshine_hours PUBLIC "/home/ftx/miniconda3/envs/qidong/include")
target_link_directories(sunshine_hours PUBLIC "/home/ftx/miniconda3/envs/qidong/lib")
target_link_libraries(sunshine_hours PUBLIC gdal CUDA::cublas CUDA::cudart "-lgomp")
