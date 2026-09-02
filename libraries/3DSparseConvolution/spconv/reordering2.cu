// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "reordering2.cu.h"
#include "reordering.h"
#include "common/launch.cuh"
#include <iostream>

namespace spconv {

// Ampere 架构 f16 稀疏卷积 Tensor Core kernel
// 对应 spconv 中 L489-L561 的调度逻辑:
//   Ampere_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_C311LLL_SK
//
// 参数说明:
//   features:        (numActIn, C)       输入特征
//   filters:         (K, kernel_volume, C) 卷积权重
//   pair_fwd:        (kernel_volume, numActOut)  输入点索引(pair_fwd)
//   pair_mask_fwd:   (numActOut,) uint32  每个输出点的mask
//   mask_argsort_fwd:(numActOut,) int32  mask argsort后的索引
//   out_features:    (numActOut, K)      输出特征
//   stream:          CUDA stream
void implicit_gemm_cuda(nv::Tensor features,
                        nv::Tensor filters,
                        nv::Tensor pair_fwd,
                        nv::Tensor pair_mask_fwd,
                        nv::Tensor mask_argsort_fwd,
                        nv::Tensor out_features,
                        void* stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    // 1. 提取维度信息
    int numActIn  = features.size(0);      // 输入有效点数
    int C         = features.size(1);      // 输入通道数
    int K         = filters.size(0);       // 输出通道数
    int kernel_volume = filters.size(1);   // 卷积核体积 (如 3x3x3=27)
    int numActOut = out_features.size(0);  // 输出有效点数
    int N = numActOut;

    // 2. 创建 ConvProblem
    ConvProblem problem(N, C, K, kernel_volume,
                        1,   // split_k_slices
                        1);  // groups

    // 3. 计算 mask_filter
    // 对于单 split, 所有 kernel 位置都活跃, mask_filter = (1 << kernel_volume) - 1
    uint32_t mask_filter = (kernel_volume >= 32) ? 0xFFFFFFFFu
                                                  : ((1u << kernel_volume) - 1);

    // 4. 获取数据指针
    const half* ptr_A = features.ptr<half>();
    const half* ptr_B = filters.ptr<half>();
    half*       ptr_C = out_features.ptr<half>();
    // ptr_D: source 指针, beta=0 时无意义, 使用 ptr_C 占位
    const half* ptr_D = ptr_C;

    const uint32_t* mask_ptr        = pair_mask_fwd.ptr<uint32_t>();
    const int*      mask_argsort_ptr = mask_argsort_fwd.ptr<int>();
    const int*      indice_ptr       = pair_fwd.ptr<int>();

    // 5. 创建 ConvParams (构造函数内部完成 grid_dims/gemm_k_iterations/迭代器参数初始化)
    ConvParams ker_params(
        problem,
        ptr_A,             // 输入特征
        ptr_B,             // 卷积权重
        ptr_C,             // 输出
        ptr_D,             // source (beta=0 时不使用)
        mask_ptr,          // per-point mask 数据
        mask_argsort_ptr,  // mask argsort 索引
        indice_ptr,        // pair_fwd 位置映射
        mask_filter,       // mask_filter: 选择活跃 kernel 位置
        false,             // reverse_mask
        __float2half(1.0),   // alpha
        __float2half(0.0),   // beta
        __float2half(0.0),   // act_alpha
        __float2half(0.0),   // act_beta
        Activation::kNone,    // act_type (cumm::conv::Activation)
        1,                 // split_k_slices
        false);            // d_is_bias

    // 6. 配置 launch 参数
    // tile_shape = {64, 128, 32}, block = 128 threads, smem = 24576 bytes
    dim3 grid  = ker_params.grid_dims;//(numActOut/64, outchannels/128, 1)
    dim3 block(128);
    int  smem_size = 24576;

    // 7. 设置共享内存属性 (>= 48KB 时需要)
    if (smem_size >= (48 << 10)) {//49152
        cudaFuncSetAttribute(
            spconv::conv_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        cudaFuncSetAttribute(
            spconv::conv_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            100);
    }

    // 8. 启动 kernel (conv_kernel 定义在 reordering2.cu.h 中)
    checkRuntime(cudaStreamSynchronize(_stream));
    std::cout << "conv_kernel begin!" << std::endl;
    std::cout << "numActIn: " << numActIn << std::endl;
    std::cout << "numActOut: " << numActOut << std::endl;
    std::cout << "conv_kernel grid dim: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    std::cout << "conv_kernel block dim: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    conv_kernel<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(ker_params);
    checkRuntime(cudaStreamSynchronize(_stream));
    std::cout << "conv_kernel end!" << std::endl;

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "implicit_gemm_cuda kernel launch failed: "
                  << cudaGetErrorString(result) << std::endl;
    }
}

} // namespace spconv
