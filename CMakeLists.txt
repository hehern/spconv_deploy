# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 2.8.12)
project(bevfusion)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

# Requires C++17
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -Wno-error=attributes -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -Wno-error=attributes -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-std=c++17 -Werror=all-warnings --diag-suppress=20012 --expt-relaxed-constexpr -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-Wno-error=attributes,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++17 --expt-relaxed-constexpr --diag-suppress=20012 -Xcompiler -std=c++17,-Wno-error=attributes -O0 -g -DENABLE_TEXT_BACKEND_STB")

# Tell nvcc's host compiler not to treat unused parameters as errors
list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler -Wno-unused-parameter -Xcompiler -Wno-error=unused-parameter)
list(APPEND CUDA_NVCC_FLAGS_DEBUG   -Xcompiler -Wno-unused-parameter)

# CUDA 架构: 优先使用环境变量 CUDASM (由 tool/environment.sh 自动探测本机 GPU),
# 探测失败时给出明确报错, 避免拼出非法的 -gencode arch=compute_
if(NOT DEFINED ENV{CUDASM} OR "$ENV{CUDASM}" STREQUAL "")
  message(FATAL_ERROR "CUDASM is empty. Run 'source tool/environment.sh' first, or set CUDASM manually (e.g. 80/86/89).")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  dependencies/stb
  dependencies/pybind11/include
  libraries/cuOSD/src
  libraries/3DSparseConvolution
  src/common
  src/
)

link_directories(
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  build
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES 
  libraries/3DSparseConvolution/normal_op/*.cu
  libraries/3DSparseConvolution/*.cpp
  libraries/3DSparseConvolution/*.cu
  libraries/3DSparseConvolution/onnx/*.cpp
  libraries/3DSparseConvolution/onnx/*.cc
  libraries/3DSparseConvolution/spconv/*.cu
  libraries/3DSparseConvolution/spconv/*.cpp
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  libraries/cuOSD/src/*.cpp
  libraries/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${CUDA_cublas_LIBRARY}
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


########################## libpybev.so ################################
if(${CMAKE_BUILD_PYTHON} STREQUAL "ON")
  cuda_add_library(pybev SHARED
    src/python.cpp
  )

  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    $ENV{Python_Soname}
  )
endif()
#######################################################################