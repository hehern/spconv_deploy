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

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <type_traits>
#include <cuda_runtime.h>
#include <iostream>
#include "spconv/indice.cu.h"
#include "spconv/indice.h"

namespace spconv {

// ===== thrust 临时缓冲内存池 =====
// thrust::sort / sort_by_key / unique 内部 (cub) 每帧会为临时存储分配 ~30 次
// cudaMalloc/cudaFree; cudaFree 隐式同步设备, 打断 kernel 流水线产生 GPU 空闲。
// 这里把 thrust 临时缓冲接到 nv::Tensor 的全局 best-fit 内存池 (tensor.cu MemoryPool),
// 帧间复用块, 稳态下临时分配零 cudaMalloc/cudaFree。
// 注意: 池非 stream-aware, 依赖本库所有 thrust 调用运行在同一推理流上
// (与 nv::Tensor 池用法一致, 同流串行保证复用安全)。
template <typename T>
struct PooledDeviceAllocator {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  template <typename U>
  struct rebind {
    using other = PooledDeviceAllocator<U>;
  };
  PooledDeviceAllocator() = default;
  template <typename U>
  PooledDeviceAllocator(const PooledDeviceAllocator<U>&) {}
  T* allocate(std::ptrdiff_t n) {
    if (n <= 0) return nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(T);
    void* p = nv::pool_acquire_device(bytes);
    if (p == nullptr) {  // 池不可用(超大/未初始化)时退回裸 cudaMalloc
      checkRuntime(cudaMalloc(&p, bytes));
    }
    return static_cast<T*>(p);
  }
  void deallocate(T* p, std::ptrdiff_t n) {
    if (n <= 0) return;
    nv::pool_release_device(p, static_cast<size_t>(n) * sizeof(T));
  }
};

// cub 排序临时缓冲: 从全局 best-fit 池取用, 析构归还 (与 thrust 池一致)。
// 排序为流上异步操作, 归还后同流复用安全。
struct CubSortTemp {
  void* ptr = nullptr;
  size_t bytes = 0;
  ~CubSortTemp() {
    if (ptr) nv::pool_release_device(ptr, bytes);
  }
  void* get(size_t need) {
    if (ptr && bytes >= need) return ptr;
    if (ptr) nv::pool_release_device(ptr, bytes);
    bytes = need;
    ptr = nv::pool_acquire_device(bytes);
    if (ptr == nullptr) checkRuntime(cudaMalloc(&ptr, bytes));  // 池不可用时退回
    return ptr;
  }
};

// 网格体积所需位数: 保证所有 voxel 1D 索引 < 2^bits (定义见 indice.h, inline)

// 限位基数排序 (cub, keys+values 就地): 排序键只需 bits 位 (voxel 索引/mask),
// 远小于 32 位, 减少 radix pass 数。cub 经典 API 需独立 in/out 缓冲, 这里用
// 池分配的备用缓冲做 DoubleBuffer, 结果若落在备用缓冲则拷回 (与 thrust 内部一致)。
// 语义与 thrust::sort_by_key 完全相同, 仅额外支持 begin/end_bit。
template <typename KeyT, typename ValueT>
static void radix_sort_pairs(KeyT* keys, ValueT* vals, int n, int bits, void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  size_t key_bytes = sizeof(KeyT) * n, val_bytes = sizeof(ValueT) * n;
  void* sk = nv::pool_acquire_device(key_bytes);
  void* sv = nv::pool_acquire_device(val_bytes);
  if (sk == nullptr) checkRuntime(cudaMalloc(&sk, key_bytes));
  if (sv == nullptr) checkRuntime(cudaMalloc(&sv, val_bytes));
  cub::DoubleBuffer<KeyT> dk(keys, static_cast<KeyT*>(sk));
  cub::DoubleBuffer<ValueT> dv(vals, static_cast<ValueT*>(sv));
  size_t temp_bytes = 0;
  checkRuntime(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, dk, dv, n, 0, bits, s));
  CubSortTemp temp;
  checkRuntime(cub::DeviceRadixSort::SortPairs(temp.get(temp_bytes), temp_bytes, dk, dv, n, 0, bits, s));
  if (dk.Current() != keys)
    checkRuntime(cudaMemcpyAsync(keys, dk.Current(), key_bytes, cudaMemcpyDeviceToDevice, s));
  if (dv.Current() != vals)
    checkRuntime(cudaMemcpyAsync(vals, dv.Current(), val_bytes, cudaMemcpyDeviceToDevice, s));
  nv::pool_release_device(sk, key_bytes);
  nv::pool_release_device(sv, val_bytes);
}


nv::Tensor find_unique_elements_cuda(nv::Tensor& src_tensor, int bits, void* stream) {

  int64_t num = src_tensor.shape[0];
  if (num == 0) {
      return nv::Tensor::create(std::vector<int64_t>{0}, nv::DataType::Int32);
  }

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int* begin = src_tensor.ptr<int>();

  // 限位基数排序 (cub, 只排 bits 位而非 32 位): voxel 索引 < 2^bits,
  // 减少 radix pass 数 (32bit=8 pass, 24bit=6 pass); INT_MAX 哨兵的高位相同,
  // 低位全 1 保证仍排到末尾, 排序语义不变。结果写入池分配的 scratch (非就位),
  // 不破坏 src_tensor 原始布局 (indice_pairs_uniq 的备份因此也不再被破坏)。
  size_t key_bytes = (size_t)num * sizeof(int);
  void* scratch = nv::pool_acquire_device(key_bytes);
  if (scratch == nullptr) checkRuntime(cudaMalloc(&scratch, key_bytes));
  size_t temp_bytes = 0;
  checkRuntime(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, begin, (int*)scratch, (int)num, 0, bits, _stream));
  CubSortTemp temp;
  checkRuntime(cub::DeviceRadixSort::SortKeys(temp.get(temp_bytes), temp_bytes, begin, (int*)scratch, (int)num, 0, bits, _stream));

  // unique 是单 pass, 保留 thrust (走池), 直接在已排序的 scratch 上做
  PooledDeviceAllocator<int> pool_alloc;
  auto policy = thrust::cuda::par(pool_alloc).on(_stream);
  int* unique_end = thrust::unique(policy, (int*)scratch, (int*)scratch + num);

  checkRuntime(cudaStreamSynchronize(_stream));

  int64_t unique_count = unique_end - (int*)scratch;
  // 必须传 _stream: from_data 的 stream 参数默认是 nullptr(默认流),
  // 否则这里 DtoD 的 cudaMemcpyAsync 会落到默认流, 破坏推理流的顺序语义
  nv::Tensor out = nv::Tensor::from_data(
    scratch, std::vector<int64_t>{unique_count}, nv::DataType::Int32, true, _stream);
  // 拷贝已在流上入队, 归还 scratch 到池 (同流复用安全)
  nv::pool_release_device(scratch, key_bytes);
  return out;
}


/*
  填充indice_pairs以及indice_pair_mask
  in:
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  hashdata_k:nv::Tensor, shape:{numAnum_voxels}
  hashdata_v:nv::Tensor, shape:{numAnum_voxels}
  indice_pairs:nv::Tensor, shape:{kernelVolume, numActIn}
  indice_num_per_loc:nv::Tensor, shape:{kernelVolume},每个卷积核元素对应的有效voxel数量
  input_dims: eg:{720, 720, 21}
  ksize: eg:{3, 3, 3}
  dilation: eg:{1, 1, 1}
  indice_pair_mask:nv::Tensor, shape:{numActIn}
*/
int generate_subm_conv_inds(nv::Tensor indices, nv::Tensor hashdata_k,
                            nv::Tensor hashdata_v, nv::Tensor indice_pairs,
                            std::vector<int> input_dims, std::vector<int> ksize,
                            nv::Tensor indice_pair_mask,
                            ConvOutLocIter& loc_iter, void* stream) {

  int numActIn = indices.shape[0];
  if (numActIn == 0) {
    return 0;
  }
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());

  int* indicesIn_ptr = indices.ptr<int>();//(batch,x,y,z)
  int* hashdata_k_ptr = hashdata_k.ptr<int>();
  int* hashdata_v_ptr = hashdata_v.ptr<int>();
  int* indice_pairs_ptr = indice_pairs.ptr<int>();
  int64_t NDim = ksize.size();//3
  nv::Tensor ou = nv::Tensor::create(std::vector<int64_t>{NDim}, nv::DataType::Int32);//output_shape
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), input_dims.data(), input_dims.size()*sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  int* inSpatialShape_ptr = ou.ptr<int>();//size:xyz
  
  cuda_linear_launch(buildSubmConvHashTable, _stream, numActIn, indicesIn_ptr, hashdata_k_ptr, hashdata_v_ptr, inSpatialShape_ptr);//计算Hash_out：建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  // hash 表按 key 升序排序 (value 跟随), 使 calc_subm_conv_indices_mask 可用二分查找。
  // 原实现为无序表 + 线性扫描, 查找复杂度 O(N^2 * RS) (subm2 层约 780 亿次比较/帧),
  // 排序 + 二分后降为 O(N log N + N * RS * log N), 是 Lidar Backbone 的主要耗时来源。
  // 限位基数排序 (只排 voxel 索引位数, 非 32 位), 备用缓冲走 best-fit 池。
  radix_sort_pairs<int, int>(hashdata_k_ptr, hashdata_v_ptr, numActIn,
                             voxel_index_bits(input_dims), stream);
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "buildSubmConvHashTable!" << std::endl;
  uint32_t* indice_pair_mask_ptr = indice_pair_mask.ptr<uint32_t>();
  cuda_linear_launch(fill_kernel<uint32_t>, _stream, numActIn, indice_pair_mask_ptr, 1 << (kv / 2));//每个active voxel都与kernel中心位置参与卷积，所以初始化1<<13
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "fill_kernel!" << std::endl;
  dim3 __threads__(std::min(numActIn, 1024));
  dim3 __blocks__(divup(numActIn, std::min(numActIn, 1024)), (kv / 2) + 1);
  calc_subm_conv_indices_mask<<<__blocks__, __threads__, 0, _stream>>>(hashdata_k_ptr, hashdata_v_ptr, indicesIn_ptr,
        indice_pairs_ptr, indice_pair_mask_ptr, numActIn, kv, (kv / 2) + 1, loc_iter);
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "calc_subm_conv_indices_mask!" << std::endl;
  return indices.shape[0];
}

nv::Tensor sort_1d_by_key_allocator_v2(nv::Tensor data,
                                       nv::Tensor indices,
                                       int bits,
                                       void* stream) {

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int numActIn = indices.numel;
  if (numActIn == 0) {
    return indices;
  }
  int32_t* indicesIn_ptr = indices.ptr<int32_t>();
  uint32_t* data_ptr = data.ptr<uint32_t>();
  cuda_linear_launch(arange_kernel<int32_t>, _stream, numActIn, indicesIn_ptr);//0-numActIn-1

  // 限位基数排序: pair_mask 只需 kernelVolume 位 (非 32), 减少 cub pass 数
  radix_sort_pairs<uint32_t, int32_t>(data_ptr, indicesIn_ptr, numActIn, bits, stream);//按照mask的大小顺序升序排列indicesIn_ptr

  return indices;
}

void generate_conv_inds_mask_stage1(nv::Tensor indices, 
                                    nv::Tensor indice_pairs_uniq,
                                    std::vector<int> ksize,
                                    ConvOutLocIter& loc_iter,
                                    void* stream) {
  
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  int num_act_in = indices.shape[0];
  if (num_act_in == 0) {
    return;
  }
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int64_t uniq_size = indice_pairs_uniq.shape[0];
  int32_t* indice_pairs_uniq_ptr = indice_pairs_uniq.ptr<int32_t>();
  cuda_linear_launch(clean_indices_uniq<int32_t>, _stream, uniq_size, indice_pairs_uniq_ptr);
  // indice_pairs_uniq.fill<int32_t>(std::numeric_limits<int32_t>::max());//这两种写法都可以

  int* indicesIn_ptr = indices.ptr<int>();//(batch,x,y,z)

  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);
  calc_conv_indices_stage1_mask<<<__blocks__, __threads__, 0, _stream>>>(indicesIn_ptr,
    indice_pairs_uniq_ptr, num_act_in, kv, loc_iter);
}

int generate_conv_inds_mask_stage2(nv::Tensor indices,
                                   nv::Tensor hashdata_k,
                                   nv::Tensor hashdata_v,
                                   nv::Tensor indice_pairs,
                                   nv::Tensor indice_pairs_uniq,
                                   nv::Tensor indice_pairs_uniq_before_sort,
                                   nv::Tensor out_inds,
                                   nv::Tensor mask_fwd,
                                   int num_out_act,
                                   std::vector<int> ksize,
                                   ConvOutLocIter& loc_iter,
                                   void* stream) {

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int num_act_in = indices.shape[0];
  if (num_act_in == 0 || num_out_act == 0) {
    return num_out_act;
  }
  int* hashdata_k_ptr = hashdata_k.ptr<int>();
  int* hashdata_v_ptr = hashdata_v.ptr<int>();
  int* indice_pairs_ptr = indice_pairs.ptr<int>();
  int* indice_pairs_uniq_ptr = indice_pairs_uniq.ptr<int>();
  int* indice_pairs_uniq_before_sort_ptr = indice_pairs_uniq_before_sort.ptr<int>();
  int* out_inds_ptr = out_inds.ptr<int>();
  uint32_t* mask_fwd_ptr = mask_fwd.ptr<uint32_t>();

  cuda_linear_launch(build_conv_hash_table, _stream, num_out_act, hashdata_k_ptr, hashdata_v_ptr, 
    out_inds_ptr, indice_pairs_uniq_ptr, loc_iter);
  
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);

  calc_conv_indices_stage2_inference_mask<<<__blocks__, __threads__, 0, _stream>>>(hashdata_k_ptr, hashdata_v_ptr, indice_pairs_ptr, 
    indice_pairs_uniq_before_sort_ptr, mask_fwd_ptr, num_act_in, num_out_act, kv, hashdata_k.shape[0]);

  return num_out_act;
}

} // namespace spconv