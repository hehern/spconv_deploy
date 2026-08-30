#pragma once
// 从 cumm 生成代码移植: out_iter_const/OutIterator.h
#include <cuda_fp16.h>
#include <vector_types.h>
#include <array>
#include <cstdint>
#include "ConvCommon.h"
#include "OutIteratorParams.h"

namespace cumm {
namespace conv {

using ConstOutThreadMap = Out5DLinear;

struct ConstOutIterator {
  const half * pointer_;
  OutIteratorParams const& params_;
  bool column_masks_[2][1];
  int32_t extent_row_;
  int32_t thread_start_row_;
  int counts_[3];
  int64_t indices_[1];
  __forceinline__ __device__  ConstOutIterator(OutIteratorParams const& params, const half * ptr, std::array<int, 2> extent, std::array<int, 2> offset_2d, int thread_idx) : params_(params)  {
    counts_[0] = 0;
    counts_[1] = 0;
    counts_[2] = 0;
    auto tmap_offset = ConstOutThreadMap::initial_offset(thread_idx);
    // thread_offset = ThreadMap::initial_offset(thread_idx) + offset_2d
    std::array<int, 2> thread_offset{tmap_offset[0] + offset_2d[0],
                                     tmap_offset[1] + offset_2d[1]};
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        for (int v = 0; v < 1; ++v){
            column_masks_[c][v] = ((thread_offset[1] + 64 * c + v * 8) < extent[1]);
        }
    }
    extent_row_ = extent[0];
    thread_start_row_ = thread_offset[0];
    #pragma unroll
    for (int cluster = 0; cluster < 1; ++cluster){
      #pragma unroll
      for (int group = 0; group < 1; ++group){
        #pragma unroll
        for (int row = 0; row < 1; ++row){
          int idx = (row +  1 * (group +  1 * cluster));
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    pointer_ = ptr + thread_offset[1];
  }
  __forceinline__ __device__ void store_with_offset(std::array<half, 16> const & frag, int32_t offset)   {

  }
  __forceinline__ __device__ void load_with_offset(std::array<half, 16>  & frag, int32_t offset)   {

    auto cur_pointer = pointer_;
    aligned_array<int4, 1, 16>  *frag_ptr = reinterpret_cast<aligned_array<int4, 1, 16>  *>(&frag);
    #pragma unroll
    for (int cluster = 0; cluster < 1; ++cluster){
      #pragma unroll
      for (int group = 0; group < 1; ++group){
        #pragma unroll
        for (int row = 0; row < 1; ++row){
          int frag_row_idx =
              (row +  1 * (group +  1 * cluster));
          // delta: [Cluster, Group, Row]
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);
          aligned_array<int4, 1, 16> const *memory_pointer =
              reinterpret_cast<aligned_array<int4, 1, 16> const *>(cur_pointer + offset + indices_[frag_row_idx]);
          #pragma unroll
          for (int column = 0; column < 2; ++column){
            bool guard = row_guard && column_masks_[column][0];
            global_load16(&frag_ptr[frag_row_idx *  2 + column],
                (const void *)&memory_pointer[column * 64 / 8],
                guard);
          }
        }
      }
    }
  }
  __forceinline__ __device__ void store(std::array<half, 16> const & frag)   {
    store_with_offset(frag, 0);
  }
  __forceinline__ __device__ void load(std::array<half, 16> & frag)   {
    load_with_offset(frag, 0);
  }
  __forceinline__ __device__ ConstOutIterator& operator++()   {
    ++counts_[2];
    // kPartShape: [Tile, Cluster, Group, Row, Col]
    thread_start_row_ += 8;
    if (counts_[2] == 4) {
    counts_[2] = 0;
    ++counts_[1];
    thread_start_row_ +=
        (2 - 1) * 8 * 4;
    if (counts_[1] == 1) {
        counts_[1] = 0;
        ++counts_[0];
        thread_start_row_ +=
            1 * 2 * 8 * 4;
        if (counts_[0] == 1) {
        counts_[0] = 0;
        }
    }
    }
    #pragma unroll
    for (int cluster = 0; cluster < 1; ++cluster){
      #pragma unroll
      for (int group = 0; group < 1; ++group){
        #pragma unroll
        for (int row = 0; row < 1; ++row){
          int idx =
              (row +  1 * (group +  1 * cluster));
          int row_offset =
              row * 4 + group * 1 + cluster * 1;
          bool row_guard = ((row_offset + thread_start_row_) < extent_row_ && params_.indice_ptr_ != nullptr);
          indices_[idx] = row_guard ? int64_t(params_.indice_ptr_[row_offset + thread_start_row_]) * int64_t(params_.stride) : 0;
        }
      }
    }
    return *this;
  }
};

} // namespace conv
} // namespace cumm
