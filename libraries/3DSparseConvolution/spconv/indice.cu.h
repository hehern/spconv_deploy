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