#include "op_add.h"
#include <cuda_fp16.h>
#include "common/launch.cuh"

namespace spconv {

__global__ void addKernel(int64_t act_num, const half *input0, const half *input1, half *output, int64_t voxel_dim) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;

  auto index = ix * voxel_dim;
  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    output[index+i] = input0[index+i] + input1[index+i];
  }
}

__global__ void addKernelV2(int64_t act_num, const half *input0, const half *input1, half *output, int64_t voxel_dim) {
  const int ELEMENTS_PER_THREAD = 8;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadsPerPosition = (voxel_dim + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int positionIdx = globalTid / threadsPerPosition;
  int laneInPosition = globalTid % threadsPerPosition;

  if (positionIdx >= act_num) return;

  int index = positionIdx * voxel_dim;
  int startPlane = laneInPosition * ELEMENTS_PER_THREAD;

  #pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    int planeIdx = startPlane + i;
    if (planeIdx < voxel_dim) {
      output[index + planeIdx] = input0[index + planeIdx] + input1[index + planeIdx];
    }
  }
}

// Add+ReLU 融合版: output = max(0, input0 + input1), 一次遍历完成,
// 省去独立 Relu kernel 的一次读写与 launch。
__global__ void addReluKernelV2(int64_t act_num, const half *input0, const half *input1, half *output, int64_t voxel_dim) {
  const int ELEMENTS_PER_THREAD = 8;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadsPerPosition = (voxel_dim + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int positionIdx = globalTid / threadsPerPosition;
  int laneInPosition = globalTid % threadsPerPosition;

  if (positionIdx >= act_num) return;

  int index = positionIdx * voxel_dim;
  int startPlane = laneInPosition * ELEMENTS_PER_THREAD;

  #pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    int planeIdx = startPlane + i;
    if (planeIdx < voxel_dim) {
      // half 三元表达式在 CUDA 11.8 有歧义, 转 float 比较
      half sum = input0[index + planeIdx] + input1[index + planeIdx];
      output[index + planeIdx] = (__half2float(sum) >= 0.0f) ? sum : half{0};
    }
  }
}


void add_cuda(nv::Tensor features0, nv::Tensor features1, nv::Tensor output,
              int64_t act_num, int64_t voxel_dim, void* stream, bool relu) {

  const half* input0_ptr = features0.ptr<half>();
  const half* input1_ptr = features1.ptr<half>();
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  // cuda_linear_launch(addKernel, _stream, act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim);

  // 空帧保护: act_num=0 时 numBlocks=0, <<<0, 256>>> 触发 invalid configuration
  if (act_num <= 0) return;

  const int ELEMENTS_PER_THREAD = 8;
  int threadsPerPosition = (voxel_dim + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int totalThreads = act_num * threadsPerPosition;
  const int blockSize = 256;
  int numBlocks = (totalThreads + blockSize - 1) / blockSize;
  if (numBlocks > 65535) numBlocks = 65535;
  dim3 blocks(numBlocks);
  dim3 threads(blockSize);
  if (relu) {
    addReluKernelV2<<<blocks, threads, 0, _stream>>>(act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim);
  } else {
    addKernelV2<<<blocks, threads, 0, _stream>>>(act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim);
  }
  // 打印输出
  // checkRuntime(cudaStreamSynchronize(_stream));
  // printf("features0 numel = %d, features1 numel = %d, act_num = %d\n", int(features0.numel), int(features1.numel), int(act_num));
  // auto outputHost = output.to_host(stream);
  // auto f_h_ptr = outputHost.ptr<half>();
  // for(size_t i=0; i<outputHost.numel; i++) {
  //   float f_f = __half2float(f_h_ptr[i]);
  //   printf("%f,", f_f);
  // }
  // printf("\n");
}


}// namespace spconv