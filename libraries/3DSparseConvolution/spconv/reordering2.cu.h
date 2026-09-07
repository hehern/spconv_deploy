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

#ifndef REORDERING2_CU_H_
#define REORDERING2_CU_H_

#include <cuda_fp16.h>
#include <mma.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "common/launch.cuh"
#include "conv/ConvParams.h"
#include "conv/BlockMmaStorage.h"
#include "conv/OutputSmemStorage.h"
#include "conv/ForwardDgradSparseIOIterator.h"
#include "conv/WeightIteratorDP4A.h"
#include "conv/MmaMultiStage.h"
#include "conv/LinearCombination.h"
#include "conv/OutIterator.h"
#include "conv/ConstOutIterator.h"
#include "conv/Output.h"

namespace spconv {

// 引入 cumm 命名空间中的类型
using cumm::conv::ConvParams;
using cumm::conv::Activation;
using cumm::conv::ConvProblem;
using cumm::conv::SparseParams;
using cumm::conv::WeightOptParams;
using cumm::conv::OutIteratorParams;
using cumm::conv::LayoutB;
using cumm::conv::gemm_smem_storage::BlockMmaStorage;
using cumm::conv::out_smem_storage::OutputSmemStorage;
using cumm::conv::ForwardDgradSparseIOIterator;
using cumm::conv::WeightIteratorDP4A;
using cumm::conv::MmaMultiStage;
using cumm::conv::LinearCombination;
using cumm::conv::OutIterator;
using cumm::conv::ConstOutIterator;
using cumm::conv::Output;

/*
    ===== conv_kernel 完整实现 =====
    搬运自: cumm/conv/main/Ampere_f16f16f16f16f16tnt_m64n128k32m32n64k32A1T1688_200_C301LLL_SK/ConvKernel/ConvKernel_conv_kernel.cu
    tile_shape = {64, 128, 32}, warp_tile = {32, 64, 32}, m为2, n为2, 所以一个block启动4个warp
    num_stage = 2, tensorop = m16n8k8, mask_sparse = true, Block (128 threads = 4 warps)
    warp分解：
        ┌─────────┬─────────┐
        │ warp(0) │ warp(2) │  warp_n=0
        │ m=0,n=0 │ m=0,n=1 │
        ├─────────┼─────────┤
        │ warp(1) │ warp(3) │  warp_n=1
        │ m=1,n=0 │ m=1,n=1 │
        └─────────┴─────────┘
        warp_m=0  warp_m=1
*/
__global__ void conv_kernel(ConvParams params) {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    extern __shared__ uint8_t SharedStorage[];
    auto gemm_shared_mem =
        reinterpret_cast<BlockMmaStorage *>(SharedStorage);
    auto out_shared_mem =
        reinterpret_cast<OutputSmemStorage *>(SharedStorage);

    // 1. Block/Thread 坐标
    int tile_offset_m = blockIdx.x;
    int tile_offset_n = blockIdx.y;//0
    int tile_offset_k = blockIdx.z;//0
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    if (tile_offset_m >= params.grid_dims.x ||
        tile_offset_n >= params.grid_dims.y) {
        return;
    }

    // 2. 计算 block 在 A/B 矩阵中的偏移
    //    A tile: 64 x 32 (M x C_reduction), 从 (numActIn, C) 输入特征 gather
    //    B tile: 128 x 32 (N x C_reduction) ← tnt: B 偏移是 {n, k}!
    //            (B = w^T 即 C x K 布局; ttt dgrad 是 {k, n})
    std::array<int, 2> block_offset_A{tile_offset_m * 64, tile_offset_k * 32};
    std::array<int, 2> block_offset_B{tile_offset_n * 128, tile_offset_k * 32};
    int thread_idx = threadIdx.x;

    // 3. 初始化稀疏输入迭代器
    //    InputIteratorA: 通过 mask+argsort 从 (numActIn, C) 输入特征 gather
    //    InputIteratorB: 读取权重 w[k, kv, c] 的转置切片
    ForwardDgradSparseIOIterator input_iter_A(
        params.itera_params_, params.problem, params.ptr_A,
        thread_idx, block_offset_A);
    WeightIteratorDP4A input_iter_B(
        params.iterb_params_, params.problem, params.ptr_B,
        thread_idx, block_offset_B);

    // 4. Warp/Thread 分解
    //    128 threads = 4 warps, 排列为 2(m) x 2(n)
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    int warp_mn = warp_idx % (2 * 2);
    int warp_idx_k = warp_idx / (2 * 2);
    int warp_m = warp_mn % 2;
    int warp_n = warp_mn / 2;

    // 5. Mask 加载与 Warp 归约 (稀疏跳过核心逻辑)
    uint32_t kmask = 0;
    std::array<uint32_t, 2> masks;
    // std::array::fill 在 CUDA 11.8 是 __host__ 函数, 设备端需用循环初始化
    #pragma unroll
    for (int i = 0; i < 2; ++i) masks[i] = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i){
        if (tile_offset_m * 64 + i * 32 + lane_idx < params.m){
            masks[i] = params.mask_ptr[tile_offset_m * 64 + i * 32 + lane_idx];
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i){
        kmask |= masks[i];
    }
    // Warp 级归约: 所有 lane 的 kmask 做 OR,由于前面每个lane计算两个不同warp的mask，所以这里做到了block级归约
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        kmask |= __shfl_xor_sync(0xffffffff, kmask, mask, 32);
    }
    // 与 mask_filter 做与运算 (选择当前 split 对应的 kernel 位置)
    kmask &= params.mask_filter;

    // 6. MMA 计算 (Tensor Core)
    //    tnt: kmask==0 时不提前 return (epilogue 仍需写0到输出)
    MmaMultiStage mma(gemm_shared_mem, thread_idx, warp_idx_k,
                          warp_m, warp_n, lane_idx);
    std::array<half, 64> accumulators;
    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 64; ++i) accumulators[i] = half{};
    if (params.gemm_k_iterations > 0){//c/32向上取整
        if (kmask != 0){//当前block内输入元素有参与conv
            mma(params.gemm_k_iterations, accumulators,
                input_iter_A, input_iter_B, accumulators,
                kmask, params.problem.kernel_volume);
        }
    }

    // 7. Epilogue: D = alpha * A@B + beta * D + activation
    LinearCombination output_op(
        params.alpha, params.beta,
        params.act_alpha, params.act_beta, params.act_type);
    std::array<int, 2> block_offset_C{tile_offset_m * 64,
                                     tile_offset_n * 128};
    std::array<int, 2> block_extent_C{params.m, params.n};
    OutIterator out_iter_C(
        params.out_params_, params.ptr_C, block_extent_C,
        block_offset_C, thread_idx);
    ConstOutIterator out_iter_source(
        params.out_params_source_, params.ptr_D, block_extent_C,
        block_offset_C, thread_idx);
    Output out(out_shared_mem, thread_idx, warp_idx_k,
                             warp_m, warp_n, lane_idx);
    out.run(output_op, accumulators, out_iter_C, out_iter_source);
#else
    printf("conv_kernel: this arch isn't supported!\n");
    assert(0);
#endif
}

} // namespace spconv
#endif
