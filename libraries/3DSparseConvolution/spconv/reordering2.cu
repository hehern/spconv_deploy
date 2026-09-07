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

// Ampere 架构 f16 稀疏卷积 Tensor Core kernel (tnt C301, kForward 前向卷积)
// 对应 spconv 中 L1763-L1809 的调度逻辑:
//   Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK
//
// GEMM 公式 (前向): out[m, k] = sum_c in[gather(m,kv), c] * w[k, kv, c]
//   A = 输入特征 (numActIn, C), gather stride = C
//   B = 权重 (K, KV, C) 的转置切片
//   输出行宽 = K
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
                        nv::Tensor bias,
                        bool relu,
                        void* stream) {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    // 空帧保护: numActOut=0 时 grid.x = div_up(0, 64) = 0,
    // conv_kernel<<<(0,1,1), 128>>> 触发 invalid configuration 直接 abort
    // (点云数量逐帧波动, 极端帧可能无有效 voxel)
    if (out_features.size(0) == 0) return;

    // 0. 通道对齐检查与 pad (修复 C 非 8 倍数时崩溃的问题)
    //    conv_kernel 全局加载路径 (ld.global.v4.b32 / cp.async.cg 16B) 是 16 字节
    //    向量访问, 要求 A/B 行内通道步长 C 为 8 个 half 的倍数:
    //      A 地址 = row * C * 2 字节                        (C=5 -> 行步长 10B, 非 16 倍数)
    //      B 地址 = (k * kv*C + kv * C + c) * 2 字节        (strides={kv*C, C} 均非 16 倍数)
    //    C 不对齐时这些访问全部触发 misaligned address, kernel 直接崩溃。
    //    修复: host 侧把 C pad 到 8 的倍数, pad 通道填 0 (0 特征 x 0 权重 = 0 贡献, 结果不变)。
    const int kCAlign = 8;
    int C_raw = features.size(1);
    int C_pad = (C_raw + kCAlign - 1) / kCAlign * kCAlign;

    if (C_pad != C_raw) {
        int numActIn_raw = features.size(0);
        int K_raw = filters.size(0);
        int kv_raw = filters.size(1);

        // pad features: (numActIn, C_raw) -> (numActIn, C_pad)
        nv::Tensor features_pad = nv::Tensor::create(
            std::vector<int64_t>{numActIn_raw, C_pad}, features.dtype(), features.device());
        features_pad.memset(0, stream);
        checkRuntime(cudaMemcpy2DAsync(
            features_pad.ptr<half>(), C_pad * sizeof(half),  // dst, dst pitch
            features.ptr<half>(), C_raw * sizeof(half),      // src, src pitch
            C_raw * sizeof(half), numActIn_raw,              // width, height
            cudaMemcpyDeviceToDevice, _stream));

        // pad filters: (K, kv, C_raw) -> (K, kv, C_pad)
        nv::Tensor filters_pad = nv::Tensor::create(
            std::vector<int64_t>{K_raw, kv_raw, C_pad}, filters.dtype(), filters.device());
        filters_pad.memset(0, stream);
        checkRuntime(cudaMemcpy2DAsync(
            filters_pad.ptr<half>(), C_pad * sizeof(half),
            filters.ptr<half>(), C_raw * sizeof(half),
            C_raw * sizeof(half), int64_t(K_raw) * kv_raw,
            cudaMemcpyDeviceToDevice, _stream));

        features = features_pad;
        filters = filters_pad;
    }

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
    // bias 融合: epilogue 的 source = bias (d_is_bias=true 时 ConstOutIterator
    // stride=0 按 K 维广播, 所有行读 bias[col]), beta=1 即 D = Accum + bias[col] + ReLU。
    // bias 为空时退回 beta=0 (不加 bias)。
    bool has_bias = !bias.empty() && bias.numel > 0;
    const half* ptr_D = has_bias ? bias.ptr<half>() : ptr_C;

    const uint32_t* mask_ptr        = pair_mask_fwd.ptr<uint32_t>();
    const int*      mask_argsort_ptr = mask_argsort_fwd.ptr<int>();
    const int*      indice_ptr       = pair_fwd.ptr<int>();

    // 5. 创建 ConvParams (构造函数内部完成 grid_dims/gemm_k_iterations/迭代器参数初始化)
    // tnt(kForward): 构造函数含 mask_out_ptr 参数 (推理传 nullptr)
    ConvParams ker_params(
        problem,
        ptr_A,             // 输入特征 (numActIn, C)
        ptr_B,             // 卷积权重 (K, KV, C)
        ptr_C,             // 输出 (numActOut, K)
        ptr_D,             // source: bias (has_bias 时) 或占位
        mask_ptr,          // per-point mask 数据
        mask_argsort_ptr,  // mask argsort 索引
        indice_ptr,        // pair_fwd 位置映射
        mask_filter,       // mask_filter: 选择活跃 kernel 位置
        false,             // reverse_mask
        __float2half(1.0),   // alpha
        __float2half(has_bias ? 1.0 : 0.0),   // beta: 加 bias
        __float2half(0.0),   // act_alpha (ReLU 不使用)
        __float2half(0.0),   // act_beta
        relu ? Activation::kReLU : Activation::kNone,  // act_type
        1,                 // split_k_slices
        has_bias);         // d_is_bias: source 按 K 维广播读 bias

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
    // std::cout << "conv_kernel begin!" << std::endl;
    // std::cout << "numActIn: " << numActIn << std::endl;
    // std::cout << "numActOut: " << numActOut << std::endl;
    // std::cout << "C(raw/pad): " << C_raw << "/" << C << ", K: " << K << ", kernel_volume: " << kernel_volume << std::endl;
    // std::cout << "conv_kernel grid dim: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
    // std::cout << "conv_kernel block dim: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    conv_kernel<<<grid, block, smem_size, reinterpret_cast<cudaStream_t>(stream)>>>(ker_params);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "implicit_gemm_cuda kernel launch failed: "
                  << cudaGetErrorString(result) << std::endl;
    }
}

} // namespace spconv
