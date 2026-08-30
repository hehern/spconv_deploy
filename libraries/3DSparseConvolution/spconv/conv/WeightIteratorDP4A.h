#pragma once
#include <cuda_fp16.h>
#include <array>
#include <cstdint>
#include "ConvProblem.h"
#include "WeightOptParams.h"
#include "inpiterb/gload/GlobalLoad.h"
#include "inpiterb/tmap/PitchLinearWarpRaked.h"

namespace cumm {
namespace conv {

using BGlobalLoad = inpiterb::gload::GlobalLoad;
using BThreadMap = inpiterb::tmap::PitchLinearWarpRaked;

struct WeightIteratorDP4A {
  WeightOptParams const& params_;
  ConvProblem const& problem_size_;
  const char * pointer_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  std::array<uint32_t, 1> mask_backup_;
  std::array<uint32_t, 1> mask_;
  __forceinline__ __device__  WeightIteratorDP4A(WeightOptParams const& params, ConvProblem const& problem_size, const half * ptr, int thread_id, const std::array<int, 2>& threadblock_offset) : params_(params), problem_size_(problem_size), pointer_(reinterpret_cast<const char *>(ptr))  {

    auto tmap_offset = BThreadMap::initial_offset(thread_id);
    // thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id)
    std::array<int, 2> thread_offset{threadblock_offset[0] + tmap_offset[0],
                                     threadblock_offset[1] + tmap_offset[1]};
    mask_.fill(0);
    reduce_channel_offset_ = thread_offset[0];
    reduce_channel_offset_backup_ = thread_offset[0];
    #pragma unroll
    for (int s = 0; s < 2; ++s){
      #pragma unroll
      for (int c = 0; c < 2; ++c){
        #pragma unroll
        for (int ss = 0; ss < 1; ++ss){
          #pragma unroll
          for (int v = 0; v < 1; ++v){
            uint32_t pred = (thread_offset[0] + s * 4 + ss < problem_size.K)
                && (thread_offset[1] + c * 64 + v * 8 < problem_size.C);
            mask_[v] |= (pred << (s * 2 + c * 1 + ss));
          }
        }
      }
    }
    pointer_ += (thread_offset[0] * params.layout.strides[0] + thread_offset[1]) * 16 / 8;
    mask_backup_ = mask_;
  }
  __forceinline__ __device__ void operator++()   {

  }
  __forceinline__ __device__ void increment_k()   {

    pointer_ += params_.inc_c;
    reduce_channel_offset_ += params_.filter_c_delta;
    #pragma unroll
    for (int s = 0; s < 2; ++s){
      #pragma unroll
      for (int ss = 0; ss < 1; ++ss){
        if (reduce_channel_offset_ + s * 4 + ss >= problem_size_.K){
            uint32_t mask = ((1u << 2) - 1) << (s * 2 + ss);
            #pragma unroll
            for (int v = 0; v < 1; ++v){
                mask_[v] = mask_[v] & (~mask);
            }
        }
      }
    }
  }
  __forceinline__ __device__ void increment_filter()   {

    pointer_ += params_.inc_rs;
  }
  __forceinline__ __device__ void increment_filter(int num)   {

    pointer_ += params_.inc_rs * num;
  }
  __forceinline__ __device__ void reset_k()   {

    pointer_ += params_.inc_c_reset;
    reduce_channel_offset_ = reduce_channel_offset_backup_;
    mask_ = mask_backup_;
  }
  __forceinline__ __device__ void clear_mask_if_not_pred(bool pred, int v)   {

    mask_[v] = pred ? mask_[v] : 0;
  }
  __forceinline__ __device__ void clear_all_mask_if_not_pred(bool pred)   {

    #pragma unroll
    for (int v = 0; v < 1; ++v){
        mask_[v] = pred ? mask_[v] : 0;
    }
  }
  __forceinline__ __device__ void clear_all_mask_if_pred(bool pred)   {

    #pragma unroll
    for (int v = 0; v < 1; ++v){
        mask_[v] = pred ? 0:  mask_[v];
    }
  }
  __forceinline__ __device__ void clear_mask_if_pred(bool pred, int v)   {

    mask_[v] = pred ? 0 : mask_[v];
  }
  __forceinline__ __device__ void add_byte_offset(int64_t byte_offset)   {

    pointer_ += byte_offset;
  }
  __forceinline__ __device__ void at()  const {

  }
  __forceinline__ __device__ bool valid(int s, int c, int ss, int v)  const {

    return mask_[v] & (1u << (s * 2 + c * 1 + ss));
  }
  __forceinline__ __device__ const std::array<half, 8> * get(int stride, int contig, int ss)  const {

    return reinterpret_cast<const std::array<half, 8> *>(pointer_ + contig * 64 * 16 / 8);
  }
  __forceinline__ __device__ void load_with_pointer_offset(std::array<half, 32>& frag, int32_t pointer_offset)   {

    frag.fill(half{});
    std::array<half, 8> *frag_ptr = reinterpret_cast<std::array<half, 8> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 2; ++s){
      #pragma unroll
      for (int c = 0; c < 2; ++c){
        #pragma unroll
        for (int ss = 0; ss < 1; ++ss){
          #pragma unroll
          for (int v = 0; v < 1; ++v){
            int idx = s * 2 +
                c * 1 + ss * 1 + v;
            std::array<half, 8> const *access_ptr = get(s, c, ss) + v + pointer_offset / 8;
            BGlobalLoad::run(frag_ptr[idx], access_ptr, valid(s, c, ss, v));
          }
        }
      }
      if (s != 1){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ const std::array<half, 8>* load_ptr_with_param(int s, int c, bool& valid_ref)   {

    std::array<half, 8> const *access_ptr = get(s, c, 0) + 0;
    valid_ref = valid(s, c, 0, 0);
    if (c == 1 && s != 1)
        pointer_ += params_.inc_strided;
    return access_ptr;
  }
  __forceinline__ __device__ void load_invalid()   {

    #pragma unroll
    for (int s = 0; s < 2; ++s){
      if (s != 1){
          pointer_ += params_.inc_strided;
      }
    }
  }
  __forceinline__ __device__ void load(std::array<half, 32>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void clear_mask()   {

  }
};
} // namespace conv
} // namespace cumm
