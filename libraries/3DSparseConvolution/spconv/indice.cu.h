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

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
// #include <cuhash/hash_table.cuh>
// #include <spconv/geometry.h>
// #include <device_atomic_functions.hpp>
#include <numeric>
#include <limits>
#include <cooperative_groups.h>
#include "common/launch.cuh"
#include "tensor.hpp"
#include "conv/ConvOutLocIter.h"

namespace spconv {

__global__ void 
buildSubmConvHashTable(size_t numActIn, const int* indicesIn, 
                       int* hash_k, int* hash_v,
                       const int* inSpatialShape) {

  int ix = cuda_linear_index;//每个active voxel分配一个thread
  if (ix >= numActIn) return;//共计分配numActIn个thread

  const auto& voxel_idx = indicesIn[4*ix+1];//indicesIn.shape = {n,4}
  const auto& voxel_idy = indicesIn[4*ix+2];
  const auto& voxel_idz = indicesIn[4*ix+3];
  int index = (voxel_idx * inSpatialShape[1] + voxel_idy) * inSpatialShape[2] + voxel_idz;//(batch_id,x,y,z) --> index
  hash_k[ix] = index;//index为从三维坐标index转换为一维index
  hash_v[ix] = ix;//填充active voxel的序号[0, numActIn-1]

}

template <typename T>
__global__ void fill_kernel(size_t numActIn, T* data, T val)   {

  int ix = cuda_linear_index;
  if (ix >= numActIn) return;
  data[ix] = T(val);
}

__global__ void calc_subm_conv_indices_mask(const int* hashdata_k, const int* hashdata_v,
                                            const int* indices_in, int32_t* indice_pairs, 
                                            uint32_t* mask, int num_indices, int RS, int RS_half,
                                            ConvOutLocIter loc_iter) {
  
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= num_indices || iy >= RS_half) return;

  int filter_offset = blockIdx.y;//0-13
  uint32_t filter_mask_out = (1u << (filter_offset % 32));
  uint32_t filter_mask_in = (1u << ((RS - 1 - filter_offset) % 32));
  loc_iter.set_filter_offset(filter_offset);

  int filter_offset_mul_indices_pair_size = filter_offset * num_indices;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * num_indices;
  // printf("ix:%d, iy:%d, num_indices:%d\n", ix, iy, num_indices);
  if (filter_offset == (RS / 2)){//kernel中心位置
    indice_pairs[filter_offset_mul_indices_pair_size + ix] = ix;
  } else {
    // 由输出坐标计算输入坐标（1.0这时候是根据输入坐标计算输出坐标）
    int nhw_offset[4];
    if (loc_iter.query_nhw(indices_in + ix * 4, nhw_offset)) {//输出坐标计算输入坐标
      auto offset = loc_iter.layout_npq(nhw_offset);//3d坐标转换为一维index
      // 二分查找: hash 表已按 key 升序排序 (见 generate_subm_conv_inds 中 sort_by_key)
      int table_offset = num_indices;
      {
        int lo = 0, hi = num_indices - 1;
        while (lo <= hi) {
          int mid = (lo + hi) >> 1;
          int key = hashdata_k[mid];
          if (key == offset) { table_offset = mid; break; }
          else if (key < offset) lo = mid + 1;
          else hi = mid - 1;
        }
      }
      if (table_offset < num_indices) {//找到的情况下
        auto input_index = hashdata_v[table_offset]; // we find a input indice idx.
        // if (input_index >= num_indices) {
        //   printf("input_index:%d, num_indices:%d\n", input_index, num_indices);
        //   assert(0);
        // }
        atomicOr(mask + ix, filter_mask_out);//或操作，将当前对应的mask位置与当前kernel_mask进行或操作
        atomicOr(mask + input_index, filter_mask_in);
        // for this output, we set correct input idx.
        indice_pairs[filter_offset_mul_indices_pair_size + ix] = input_index;
  
        // the output in "input location" connect this output idx in another location.
        indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = ix;
      }
    }
    
  }
}

template <typename T>
__global__ void arange_kernel(size_t numActIn, T* data)   {
  
  int ix = cuda_linear_index;
  if (ix >= numActIn) return;
  data[ix] = T(ix);
}

// ===== direct_table (线性哈希表) 相关 =====
// 取代 sort+unique+二分查找 构建 stride 规则簿:
//   stage1 把输出 voxel 1D 索引哈希插入(原子CAS线性探测, 天然去重, 无需排序),
//   unique_hash 收集表项并赋输出id, stage2 O(1) 查表 (替代 find_in_hash_k 二分)。
// 参考 spconv 2.x 的 direct_table 分支 (LinearHashTableSplit + insert_key_only /
// lookup_offset / arange_hash_table)。
template <typename K = int, typename V = int>
struct LinearHashTable {
  K* key_ptr_;
  V* value_ptr_;
  int hash_size_;
  static constexpr K empty_key = std::numeric_limits<K>::max();  // voxel 索引 < 类型最大值

  __device__ int hash_slot(K key) const {
    // 乘法散列 + 取模 (hash_size 非 2 幂); 插/查必须一致
    unsigned int h = (unsigned int)key * 2654435761u;
    return (int)(h % (unsigned int)hash_size_);
  }
  // 只插 key (去重): 成功(写入)或 key 已存在都算完成
  __device__ void insert_key_only(K key) {
    int slot = hash_slot(key);
    unsigned int key_u = (unsigned int)key;
    while (true) {
      unsigned int prev = atomicCAS((unsigned int*)&key_ptr_[slot], (unsigned int)empty_key, key_u);
      if (prev == (unsigned int)empty_key || prev == key_u) return;
      slot = (slot + 1 == hash_size_) ? 0 : slot + 1;
    }
  }
  // 插 key+value: 首个写入者生效 (key 已存在时本线程的 value 覆盖同值, 键唯一场景无影响)
  __device__ void insert(K key, V value) {
    int slot = hash_slot(key);
    unsigned int key_u = (unsigned int)key;
    while (true) {
      unsigned int prev = atomicCAS((unsigned int*)&key_ptr_[slot], (unsigned int)empty_key, key_u);
      if (prev == (unsigned int)empty_key || prev == key_u) {
        value_ptr_[slot] = value;
        return;
      }
      slot = (slot + 1 == hash_size_) ? 0 : slot + 1;
    }
  }
  // 返回 key 所在槽, 不存在返回 -1 (O(1) 均摊)
  __device__ int lookup_offset(K key) const {
    int slot = hash_slot(key);
    while (true) {
      K k = key_ptr_[slot];
      if (k == key) return slot;
      if (k == empty_key) return -1;
      slot = (slot + 1 == hash_size_) ? 0 : slot + 1;
    }
  }
};

// subm 路径使用的 uint 哈希表 (voxel 索引/值均为 unsigned)
template <typename K, typename V>
using HashTable = LinearHashTable<K, V>;

// Conv3DProblem: 设备可拷贝的 conv 参数 (kernel 内据此构造 ConvOutLocIter)
struct Conv3DProblem {
  int ksize[3];
  int stride[3];
  int padding[3];
  int dilation[3];
  int output_dims[3];
  int input_dims[3];
};

// stage1 direct_table: 计算输出 voxel 1D 索引 -> 哈希插入(去重) + 写入 (kv,input) 布局数组
// (该数组即 stage2 的 "before_sort" 输入, 与旧 sort 方案的 indice_pairs_uniq 同布局)
__global__ void calc_conv_indices_stage1_mask_direct_table(
    LinearHashTable<int,int> table, const int* indices_in,
    int* indice_pairs_for_uniq, int num_indices_in, int RS,
    ConvOutLocIter loc_iter) {
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= num_indices_in || iy >= RS) return;
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  int filter_offset_mul = filter_offset * num_indices_in;
  int npq_offset[4];
  if (loc_iter.query_npq(indices_in + ix * 4, npq_offset)) {
    int index = loc_iter.layout_npq(npq_offset);
    table.insert_key_only(index);  // 去重插入
    indice_pairs_for_uniq[filter_offset_mul + ix] = index;
  }
}

// unique_hash: 遍历表, 置位槽赋顺序输出id, 收集 key 到 out_indices_offset, count=总数
// (表遍历序=任意序, 无需求升序 —— stage2 走哈希 O(1) 查找)
__global__ void arange_hash_table_kernel(long num, LinearHashTable<int,int> table,
                                         int* out_indices_offset, int* count, int limit) {
  int i = cuda_linear_index;
  if (i >= num) return;
  int key = table.key_ptr_[i];
  if (key != LinearHashTable<int,int>::empty_key) {
    int output_index = atomicAdd(count, 1);
    table.value_ptr_[i] = output_index < limit ? output_index : -1;
    if (output_index < limit) out_indices_offset[output_index] = key;
  }
}

// 映射 host 内存里的计数发布槽: 设备端 publish 内核写, host 侧只读轮询。
// 语义见 unique_hash_table 的注释 (零拷贝 producer-consumer 模式)。
struct PinnedCount { int version; int count; };

// 发布计数到映射 host 内存 (cudaHostAllocMapped)。先写 count 并 __threadfence_system()
// 保证对 host 可见, 再写 version; host 轮询到 version 变化即可安全读 count。
__global__ void publish_count_kernel(PinnedCount* out, const int* src, int version) {
  if (threadIdx.x == 0) {
    out->count = *src;
    __threadfence_system();
    out->version = version;
  }
}

// 把 unique 输出 1D 索引解码成 (batch,x,y,z) 坐标
__global__ void assign_out_indices_kernel(long num, int* indices_out, const int* out_indices_offset,
                                          ConvOutLocIter loc_iter) {
  int i = cuda_linear_index;
  if (i >= num) return;
  loc_iter.inverse(out_indices_offset[i], indices_out + 4 * i);
}

// stage2 direct_table: O(1) 查表替代二分查找, 填 mask_fwd + indice_pairs_fwd
__global__ void calc_conv_indices_stage2_mask_direct_table(
    LinearHashTable<int,int> table, int* indice_pairs_fwd,
    const int* indice_pairs_uniq_before_sort, uint32_t* mask_fwd,
    int num_indices_in, int num_indices_out, int RS) {
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= num_indices_in || iy >= RS) return;
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_fwd = (1u << (filter_offset % 32));
  auto indice_pairs_filter = indice_pairs_fwd + filter_offset * num_indices_out;
  auto bkp_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;
  int output_coord_offset = bkp_filter[ix];
  if (output_coord_offset != LinearHashTable<int,int>::empty_key) {
    int table_offset = table.lookup_offset(output_coord_offset);
    if (table_offset != -1) {
      int output_index = table.value_ptr_[table_offset];
      atomicOr(mask_fwd + output_index, filter_mask_fwd);
      indice_pairs_filter[output_index] = ix;
    }
  }
}

// ===== subm 规则簿: 单 cooperative kernel (三阶段, 取代排序+二分) =====
// 取代 buildSubmConvHashTable + radix sort_by_key + fill_kernel + calc_subm_conv_indices_mask:
// 清表 -> grid.sync -> 建哈希(key=voxel 1D 索引, value=voxel id)+mask 初始化
// -> grid.sync -> O(1) 查邻居填 mask/pairs (二分->哈希)。
// 注意: 必须经 cudaLaunchCooperativeKernel 启动 (grid.sync 要求整 grid 驻留)。
__global__ void calc_subm_conv_inds_coop_kernel(
    LinearHashTable<int,int> table, const int* indices_in, int32_t* indice_pairs,
    uint32_t* mask, int num_indices, int RS, int RS_half,
    int d0, int d1, int d2, ConvOutLocIter loc_iter) {
  namespace cg = cooperative_groups;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  // Phase 0: 清哈希表 (池复用的块有脏数据, 必须清成空 key)
  for (int i = tid; i < table.hash_size_; i += nthreads) {
    table.key_ptr_[i] = LinearHashTable<int,int>::empty_key;
  }
  cg::this_grid().sync();

  // Phase 1: 每 voxel 建哈希 (key=voxel 1D 索引, value=voxel id) + mask 初始化
  for (int ix = tid; ix < num_indices; ix += nthreads) {
    const int* idx = indices_in + ix * 4;  // (batch, x, y, z)
    int index = (idx[1] * d1 + idx[2]) * d2 + idx[3];
    table.insert(index, ix);
    mask[ix] = 1u << (RS / 2);  // 中心 kernel 位 (与原 fill_kernel 的 1<<(kv/2) 一致)
  }
  cg::this_grid().sync();

  // Phase 2: 每 (filter, voxel) O(1) 查表 (原 calc_subm_conv_indices_mask, 二分->哈希)
  for (int filter_offset = 0; filter_offset < RS_half; ++filter_offset) {
    uint32_t filter_mask_out = (1u << (filter_offset % 32));
    uint32_t filter_mask_in = (1u << ((RS - 1 - filter_offset) % 32));
    loc_iter.set_filter_offset(filter_offset);
    int f_mul = filter_offset * num_indices;
    int f_mul_1 = (RS - 1 - filter_offset) * num_indices;
    bool is_center = (filter_offset == (RS / 2));
    for (int ix = tid; ix < num_indices; ix += nthreads) {
      if (is_center) {  // kernel 中心位置: 自己连自己
        indice_pairs[f_mul + ix] = ix;
        continue;
      }
      int nhw_offset[4];
      if (loc_iter.query_nhw(indices_in + ix * 4, nhw_offset)) {  // 邻居坐标
        auto offset = loc_iter.layout_npq(nhw_offset);             // 邻居 1D 索引
        int table_offset = table.lookup_offset(offset);            // O(1) 查表
        if (table_offset != -1) {
          auto input_index = table.value_ptr_[table_offset];
          atomicOr(mask + ix, filter_mask_out);
          atomicOr(mask + input_index, filter_mask_in);
          indice_pairs[f_mul + ix] = input_index;
          indice_pairs[f_mul_1 + input_index] = ix;
        }
      }
    }
  }
}

template <typename T>
__global__ void clean_indices_uniq(size_t size, T* indice_pairs_for_uniq)   {
  
  int ix = cuda_linear_index;
  if (ix >= size) return;
  indice_pairs_for_uniq[ix] = std::numeric_limits<T>::max();
}

__global__ void calc_conv_indices_stage1_mask(const int* indices_in, 
                                              int32_t* indice_pairs_for_uniq, 
                                              int num_indices_in, int RS,
                                              ConvOutLocIter loc_iter) {
  
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= num_indices_in || iy >= RS) return;

  int filter_offset = blockIdx.y;//0-RS
  loc_iter.set_filter_offset(filter_offset);
  int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;

  int npq_offset[4];
  if (loc_iter.query_npq(indices_in + ix * 4, npq_offset)) {//计算输出坐标
    auto index = loc_iter.layout_npq(npq_offset);
    indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + ix] = index;
  }

}

__global__ void build_conv_hash_table(size_t numAct,
                                      int* hashdata_k, int* hashdata_v,
                                      int* indices_out, int* indice_pairs_for_uniq, 
                                      ConvOutLocIter loc_iter) {
  int ix = cuda_linear_index;
  if (ix >= numAct) return;

  auto output_coord_offset = indice_pairs_for_uniq[ix];
  loc_iter.inverse(output_coord_offset, indices_out + 4 * ix);
  hashdata_k[ix] = output_coord_offset;
  hashdata_v[ix] = ix;
}

// 二分查找: hash_k 来自 find_unique_elements_cuda (sort+unique) 拷贝, 天然升序唯一,
// 原 O(N) 线性扫描被 calc_conv_indices_stage2_inference_mask 每线程调用, 整体 O(N^2*RS)。
__device__ int find_in_hash_k(const int* hash_k, int hash_size, int value) {
  int lo = 0, hi = hash_size - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    int key = hash_k[mid];
    if (key == value) return mid;
    else if (key < value) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

__global__ void calc_conv_indices_stage2_inference_mask(int* hash_k, int* hash_v, 
                                                        int* indice_pairs,
                                                        const int* indice_pairs_uniq_before_sort, 
                                                        uint32_t* mask_fwd, 
                                                        int num_indices_in, 
                                                        int num_indices_out,
                                                        int RS, int hash_size) {
  
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  if (ix >= num_indices_in || iy >= RS) return;

  int filter_offset = blockIdx.y;
  uint32_t filter_mask_fwd = (1u << (filter_offset % 32));
  auto indice_pairs_filter = indice_pairs + filter_offset * num_indices_out;

  auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;
  auto output_coord_offset = indice_pairs_uniq_before_sort_filter[ix];

  if (output_coord_offset != std::numeric_limits<int32_t>::max()){
    auto table_offset = find_in_hash_k(hash_k, hash_size, output_coord_offset);
    if (table_offset != -1){//找到的情况下
      auto output_index = hash_v[table_offset];
      atomicOr(mask_fwd + output_index, filter_mask_fwd);
      indice_pairs_filter[output_index] = ix;
    }
  }
}


} // namespace spconv
#endif