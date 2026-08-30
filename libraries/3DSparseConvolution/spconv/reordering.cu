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
#include "cutlass/gemm/device/gemm_universal.h"

#include "spconv/reordering.cu.h"
#include "spconv/reordering.h"
#include "common/launch.cuh"
#include <iostream>
#include <cublas_v2.h>

namespace spconv {

// CUTLASS 4.6 + Sm80 + OpClassTensorOp + fp16
// 参考: cutlass/examples/47_ampere_gemm_universal_streamk
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  half alpha,
  half const *A,
  int lda,
  half const *B,
  int ldb,
  half beta,
  half *C,
  int ldc,
  cudaStream_t stream = nullptr) {

  using RowMajor = cutlass::layout::RowMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;

  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  constexpr int NumStages = 4;
  constexpr int AlignmentA = 8;
  constexpr int AlignmentB = 8;
  constexpr int AlignmentC = 8;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, AlignmentC, ElementAccumulator, ElementCompute>;

  using CutlassGemm = cutlass::gemm::device::GemmUniversal<
      ElementA, RowMajor, ElementB, RowMajor, ElementC, RowMajor,
      ElementAccumulator, OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      NumStages, AlignmentA, AlignmentB>;

  CutlassGemm gemm_operator;

  int64_t batch_stride_A = static_cast<int64_t>(M) * K;
  int64_t batch_stride_B = static_cast<int64_t>(K) * N;
  int64_t batch_stride_C = static_cast<int64_t>(M) * N;
  int64_t batch_stride_D = batch_stride_C;

  typename CutlassGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},
      1,
      {ElementCompute(alpha), ElementCompute(beta)},
      A, B, C, C,
      batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D,
      lda, ldb, ldc, ldc);

  cutlass::Status status = gemm_operator.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[CUTLASS] can_implement failed: %d (M=%d N=%d K=%d)\n",
            int(status), M, N, K);
    return cudaErrorUnknown;
  }

  status = gemm_operator.initialize(args, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[CUTLASS] initialize failed: %d\n", int(status));
    return cudaErrorUnknown;
  }

  status = gemm_operator(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "[CUTLASS] run failed: %d\n", int(status));
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

void matrix_multiply_cuda(const nv::Tensor& features, const nv::Tensor& filters, nv::Tensor& output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream) {
  half* features_ptr = features.ptr<half>();
  half* output_ptr = output.ptr<half>();
  matrix_multiply_cuda(features_ptr, filters, output_ptr, numActOut, numOutPlanes, numInPlanes, filter_offset, stream);
}

void matrix_multiply_cuda(half* features, const nv::Tensor& filters, half* output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream) {
  half* weight_ptr = filters.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  if (numInPlanes >= 16) {
    // K >= 16: 使用 CUTLASS TensorCore
    CutlassSgemmNN(numActOut, numOutPlanes, numInPlanes, __float2half(1.0f),
        features, numInPlanes,
        weight_ptr + filter_offset * numInPlanes * numOutPlanes, numOutPlanes,
        __float2half(0.0f), output, numOutPlanes,
        _stream);
  } else {
    // K < 16: 回退到手写 WMMA kernel
    constexpr int WMMA_M=16;
    constexpr int WMMA_N=16;
    constexpr int WMMA_K=16;
    constexpr int WMMA_TILE_M=4;
    constexpr int WMMA_TILE_N=2;
    dim3 __threads__(32 * WMMA_TILE_M * WMMA_TILE_N);
    dim3 __blocks__(divup(numOutPlanes, WMMA_N*WMMA_TILE_N), divup(numActOut, WMMA_M*WMMA_TILE_M));
    hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M,WMMA_N,WMMA_K,WMMA_TILE_M,WMMA_TILE_N><<<__blocks__, __threads__, 0, _stream>>>(
        features, weight_ptr + filter_offset * numInPlanes * numOutPlanes, output,
        numActOut, numOutPlanes, numInPlanes);
  }
}

/***
 * buffer: (count_max_size, 5)输出缓存区，需要计算的量，等下在函数中填充对应voxel的特征值
 * features: 输入特征(numActIn,5),5为特征维度，也可能是16、32等
 * indices: shape:{2,27,numActIn},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即[0, numActOut-1]
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
void sparse_gather_cuda(nv::Tensor& buffer, const nv::Tensor& features,
                        const nv::Tensor& indices, int size, int indice_offset, 
                        void* stream) {
  if (size <= 0)//当前kernel元素位置没有参数计算
    return;
  int numPlanes = features.size(1);//eg:5
  int num_act = features.size(0);//输入active voxel数量
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* features_ptr = features.ptr<half>();
  int* indices_ptr = indices.ptr<int>();
  // 将当前conv kernel元素对应的所有输入active voxel的特征值从features中取出，放到buffer中，等下在matrix_multiply_cuda中进行矩阵乘法计算
  // cuda_linear_launch(gatherGenericKernel, _stream, size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes, num_act);
  // const int BM = 128;
  // const int BK = 4;
  // dim3 __threads__(BM);
  // dim3 __blocks__(divup(size, BM));
  // gatherGenericKernelV2<BM, BK><<<__blocks__, __threads__, 0, _stream>>>(size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes);
  cuda_linear_launch(gatherGenericKernelV3, _stream, size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes);
}

void sparse_scatter_add_cuda(const nv::Tensor& buffer, nv::Tensor& outFeatures,
                             const nv::Tensor& indices, int size, int indice_offset,
                             void* stream) {
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* outFeatures_ptr = outFeatures.ptr<half>();
  int* indices_ptr = indices.ptr<int>();
  // cuda_linear_launch(scatterAddGenericKernel, _stream, size, outFeatures_ptr, buffer_ptr, indices_ptr+indice_offset, numPlanes);
  cuda_linear_launch(scatterAddGenericKernelV2, _stream, size, outFeatures_ptr, buffer_ptr, indices_ptr+indice_offset, numPlanes);

}

void addBiasAndRelu(nv::Tensor features, nv::Tensor bias,
                    bool Relu, void* stream) {
  int num_act = features.size(0);
  int numPlanes = features.size(1);//eg:16
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* features_ptr = features.ptr<half>();
  half* bias_ptr = bias.ptr<half>();
  // cuda_linear_launch(addBiasAndReluKernel, _stream, num_act, features_ptr, bias_ptr, numPlanes, Relu);
  
  // 按numPlanes调整grid: 每线程处理8个plane
  const int ELEMENTS_PER_THREAD = 8;
  int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int totalThreads = num_act * threadsPerPosition;
  const int blockSize = 256;
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;
  if (numBlocks > 65535) numBlocks = 65535;
  dim3 blocks(numBlocks);
  dim3 threads(blockSize);
  addBiasAndReluKernelV2<<<blocks, threads, 0, _stream>>>(num_act, features_ptr, bias_ptr, numPlanes, Relu);

}

/************************************************************************
 * 批量处理实现 - 将27次循环合并为单次操作
 ************************************************************************/

void sparse_gather_all_cuda(nv::Tensor& buffer, const nv::Tensor& features,
                            const nv::Tensor& indices, const int* kernelIds,
                            const int* kernelOffsets,
                            int numActIn, int totalCount, void* stream,
                            int kernelVolume) {
  if (totalCount <= 0) return;

  int numPlanes = features.size(1);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* features_ptr = features.ptr<half>();
  int* indices_ptr = indices.ptr<int>();

  // cuda_linear_launch(gatherAllKernel, _stream, totalCount, buffer_ptr, features_ptr, indices_ptr, kernelOffsets, kernelVolume, numPlanes, numActIn);
  // cuda_linear_launch(gatherAllKernelV2, _stream, totalCount, buffer_ptr, features_ptr, indices_ptr, kernelIds, kernelOffsets, numActIn, numPlanes);
  // return;

  // 根据 numPlanes 选择不同的 grid 配置
  const int blockSize = 256;
  int numBlocks;
  
  if (numPlanes <= 8) {
    // 策略A：每个线程处理一个 position
    numBlocks = (totalCount + blockSize - 1) / blockSize;
  } else {
    // 策略B：每个线程处理 8 个元素
    const int ELEMENTS_PER_THREAD = 8;
    int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    int totalThreads = totalCount * threadsPerPosition;
    numBlocks = (totalThreads + blockSize - 1) / blockSize;
  }
  // printf("sparse_gather_all_cuda: totalCount=%d, numPlanes=%d, blockSize=%d, numBlocks=%d\n", totalCount, numPlanes, blockSize, numBlocks);
  
  // 限制 grid size
  if (numBlocks > 65535) numBlocks = 65535;
  
  dim3 blocks(numBlocks);
  dim3 threads(blockSize);
  
  gatherAllKernelUnified<<<blocks, threads, 0, _stream>>>(
      totalCount, buffer_ptr, features_ptr, indices_ptr, kernelIds, kernelOffsets, numActIn, numPlanes);
  
}

void sparse_scatter_add_all_cuda(nv::Tensor& buffer, nv::Tensor& output,
                                 const nv::Tensor& indices, const int* kernelIds,
                                 const int* kernelOffsets,
                                 int numActIn, int totalCount, void* stream,
                                 int kernelVolume) {
  if (totalCount <= 0) return;

  int numPlanes = output.size(1);
  int numActOut = output.size(0);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* output_ptr = output.ptr<half>();
  int* indices_ptr = indices.ptr<int>();

  // cuda_linear_launch(scatterAddAllKernel, _stream, totalCount, output_ptr, buffer_ptr, indices_ptr, kernelOffsets, kernelVolume, numPlanes, numActIn);
  // cuda_linear_launch(scatterAddAllKernelV2, _stream, totalCount, output_ptr, buffer_ptr, indices_ptr, kernelIds, kernelOffsets, numActIn, numPlanes, numActOut, kernelVolume);

  // 根据 numPlanes 选择不同的 grid 配置
  const int blockSize = 256;
  int numBlocks;

  // 策略：每线程处理 8 个元素
  const int ELEMENTS_PER_THREAD = 8;
  int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int totalThreads = totalCount * threadsPerPosition;
  numBlocks = (totalThreads + blockSize - 1) / blockSize;

  if (numBlocks > 65535) numBlocks = 65535;

  dim3 blocks(numBlocks);
  dim3 threads(blockSize);

  scatterAddAllKernelV3<<<blocks, threads, 0, _stream>>>(
      totalCount, output_ptr, buffer_ptr, indices_ptr, kernelIds, kernelOffsets, numActIn, numPlanes, numActOut, kernelVolume);
}

} // namespace spconv
