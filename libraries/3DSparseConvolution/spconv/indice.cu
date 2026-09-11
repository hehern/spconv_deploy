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
// ===== generate_subm_conv_inds: 单 cooperative kernel 构建 subm 规则簿 =====
// 见 indice.cu.h 的 calc_subm_conv_inds_coop_kernel。哈希 O(1) 查表替代排序+二分;
// hash_k/v 由调用方按 >= 2*numActIn 分配 (哈希表, 负载<=0.5)。
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
  int RS_half = kv / 2 + 1;

  LinearHashTable<int,int> table{hashdata_k.ptr<int>(), hashdata_v.ptr<int>(), (int)hashdata_k.shape[0]};
  const int* indices_in = indices.ptr<int>();
  int32_t* pairs = indice_pairs.ptr<int32_t>();
  uint32_t* mask = indice_pair_mask.ptr<uint32_t>();
  int d0 = input_dims[0], d1 = input_dims[1], d2 = input_dims[2];

  // cooperative launch 要求整个 grid 同时驻留: 按 SM 数 x 每 SM 可驻留 block 数定 grid
  int threads = 256;
  int device = 0;
  checkRuntime(cudaGetDevice(&device));
  int num_sms = 0;
  checkRuntime(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  checkRuntime(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, calc_subm_conv_inds_coop_kernel, threads, 0));
  int grid_blocks = std::max(1, num_sms * blocks_per_sm);

  void* args[] = {(void*)&table, (void*)&indices_in, (void*)&pairs, (void*)&mask,
                  (void*)&numActIn, (void*)&kv, (void*)&RS_half,
                  (void*)&d0, (void*)&d1, (void*)&d2, (void*)&loc_iter};
  checkRuntime(cudaLaunchCooperativeKernel((void*)calc_subm_conv_inds_coop_kernel,
      dim3(grid_blocks), dim3(threads), (void**)args, 0, _stream));
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

// ===== direct_table (线性哈希表) 相关 host 包装 =====
// 取代 sort+unique+二分: 见 indice.cu.h 的 LinearHashTable 与 kernel 说明。

// stage1: 清哈希表(空key) + 清 (kv,input) 数组 + 插入去重
void generate_conv_inds_mask_stage1_direct_table(nv::Tensor indices,
                                                 nv::Tensor hash_k, nv::Tensor hash_v,
                                                 nv::Tensor indice_pairs_uniq_bkp,
                                                 std::vector<int> ksize,
                                                 ConvOutLocIter& loc_iter,
                                                 void* stream) {
  int num_act_in = (int)indices.shape[0];
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int hash_size = (int)hash_k.shape[0];
  cuda_linear_launch(clean_indices_uniq<int32_t>, _stream, hash_size, hash_k.ptr<int32_t>());
  cuda_linear_launch(clean_indices_uniq<int32_t>, _stream, indice_pairs_uniq_bkp.shape[0], indice_pairs_uniq_bkp.ptr<int32_t>());

  LinearHashTable<int,int> table{hash_k.ptr<int>(), hash_v.ptr<int>(), hash_size};
  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);
  calc_conv_indices_stage1_mask_direct_table<<<__blocks__, __threads__, 0, _stream>>>(
      table, indices.ptr<int>(), indice_pairs_uniq_bkp.ptr<int>(), num_act_in, kv, loc_iter);
  checkRuntime(cudaGetLastError());
}

// unique_hash: 收集表项 -> indice_pairs_uniq (按输出id), 返回 num_act_out (host 读回)
int unique_hash_table(nv::Tensor hash_k, nv::Tensor hash_v,
                      nv::Tensor indice_pairs_uniq, void* stream) {
  int hash_size = (int)hash_k.shape[0];
  int limit = (int)indice_pairs_uniq.shape[0];
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  nv::Tensor cnt = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, true);
  cnt.fill<int>(0, stream);
  LinearHashTable<int,int> table{hash_k.ptr<int>(), hash_v.ptr<int>(), hash_size};
  cuda_linear_launch(arange_hash_table_kernel, _stream, hash_size, table,
                     indice_pairs_uniq.ptr<int>(), cnt.ptr<int>(), limit);
  // 计数回读: 零拷贝 producer-consumer, 替代 to_host() 的 cudaMemcpyAsync +
  // cudaStreamSynchronize (驱动唤醒 ~5-10us, 且语义上排空整条流)。
  // 关键点: 发布用 设备端 publish 内核写 cudaHostAllocMapped 映射内存, host 侧
  // 轮询版本号。host 从不写映射缓冲 (版本号由设备 cudaMemset 初始化 + 每次 publish
  // 递增), 避免 CPU 缓存脏行导致设备写不可见 —— 之前用 cudaHostAllocPortable +
  // DtoH memcpy + host 写 -1 哨兵自旋, 实测被卡住 ~300us (拷贝引擎的 DMA 写不会
  // 及时失效 host 缓存的哨兵)。版本号单调递增, 也避免读到上一次的陈旧计数。
  static PinnedCount* h_pc = nullptr;   // host 视图
  static PinnedCount* d_pc = nullptr;   // device 视图 (UVA)
  static int seq_ = 0;
  if (h_pc == nullptr) {
    checkRuntime(cudaHostAlloc(&h_pc, sizeof(PinnedCount), cudaHostAllocMapped));
    checkRuntime(cudaHostGetDevicePointer(&d_pc, h_pc, 0));
    checkRuntime(cudaMemset(d_pc, 0, sizeof(PinnedCount)));  // 设备初始化 version=0/count=0
  }
  int expect = ++seq_;  // 本次调用的版本号 (host 已知, 单调递增)
  publish_count_kernel<<<1, 1, 0, _stream>>>(d_pc, cnt.ptr<int>(), expect);
  while (*static_cast<volatile int*>(&h_pc->version) != expect) { /* 自旋等 publish 落定 */ }
  int num_act_out = h_pc->count;
  return num_act_out;
}

// 解码 unique 输出 1D 索引 -> out_inds 坐标
void assign_output_direct_hash(nv::Tensor out_inds, nv::Tensor indice_pairs_uniq,
                               int num_act_out, ConvOutLocIter& loc_iter,
                               void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_linear_launch(assign_out_indices_kernel, _stream, num_act_out,
                     out_inds.ptr<int>(), indice_pairs_uniq.ptr<int>(), loc_iter);
}

// stage2: O(1) 查表填 mask_fwd + indice_pairs_fwd
void generate_conv_inds_stage2_mask_direct_table(nv::Tensor indices,
                                                 nv::Tensor hash_k, nv::Tensor hash_v,
                                                 nv::Tensor indice_pairs,
                                                 nv::Tensor indice_pairs_uniq_bkp,
                                                 nv::Tensor mask_fwd,
                                                 std::vector<int> ksize,
                                                 void* stream) {
  int num_act_in = (int)indices.shape[0];
  int num_act_out = (int)mask_fwd.shape[0];
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int hash_size = (int)hash_k.shape[0];
  LinearHashTable<int,int> table{hash_k.ptr<int>(), hash_v.ptr<int>(), hash_size};
  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);
  calc_conv_indices_stage2_mask_direct_table<<<__blocks__, __threads__, 0, _stream>>>(
      table, indice_pairs.ptr<int>(), indice_pairs_uniq_bkp.ptr<int>(),
      mask_fwd.ptr<uint32_t>(), num_act_in, num_act_out, kv);
  checkRuntime(cudaGetLastError());
}

} // namespace spconv