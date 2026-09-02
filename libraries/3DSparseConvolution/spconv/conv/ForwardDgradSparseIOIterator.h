#pragma once
#include <cuda_fp16.h>
#include <vector_types.h>
#include <array>
#include <cstdint>
#include "inpitera/mask/Mask.h"
#include "inpitera/gload/GlobalLoad.h"
#include "inpitera/tmap/PitchLinearWarpRaked.h"
#include "ConvCommon.h"
#include "ConvProblem.h"
#include "SparseParams.h"
namespace cumm {
namespace conv {
using Mask = inpitera::mask::Mask;
using GlobalLoad = inpitera::gload::GlobalLoad;
using ThreadMap = inpitera::tmap::PitchLinearWarpRaked;
struct ForwardDgradSparseIOIterator {
  SparseParams & params_;
  ConvProblem const& problem_;
  const char * pointer_;
  int const* indice_ptr_;
  int reduce_channel_offset_;
  int reduce_channel_offset_backup_;
  std::array<uint32_t, 1> mask_reset_backup_;
  std::array<uint32_t, 1> mask_;
  int32_t indices_[2];
  __forceinline__ __device__  ForwardDgradSparseIOIterator(SparseParams & params, ConvProblem const& problem_size, const half * ptr, int thread_id, const std::array<int, 2>& threadblock_offset) : params_(params), problem_(problem_size), indice_ptr_(params.indice_ptr_)  {

    auto tmap_offset = ThreadMap::initial_offset(thread_id);
    // thread_offset = threadblock_offset + ThreadMap::initial_offset(thread_id)
    std::array<int, 2> thread_offset{threadblock_offset[0] + tmap_offset[0],
                                     threadblock_offset[1] + tmap_offset[1]};
    int stride_offset_ = thread_offset[0];
    pointer_ = reinterpret_cast<const char *>(ptr + thread_offset[1]);
    params.mask_argsort_ptr_ += stride_offset_;
    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 1; ++i) mask_[i] = 0;
    reduce_channel_offset_ = thread_offset[1];
    reduce_channel_offset_backup_ = thread_offset[1];
    #pragma unroll
    for (int s = 0; s < 2; ++s){
        #pragma unroll
        for (int ss = 0; ss < 1; ++ss){
            #pragma unroll
            for (int v = 0; v < 1; ++v){
                uint32_t pred = (stride_offset_ + s * 8 + ss) < problem_.N;
                mask_[v] |= (pred << (s * 1 + ss));
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < 1; ++v){
        // 前向卷积: A 操作数是输入 features, 每行 C 个通道; 通道边界按 problem_.C 判断
        mask_[v] = thread_offset[1] + v * 8 >= problem_.C ? 0 : mask_[v];
    }
    mask_reset_backup_ = mask_;
  }
  __forceinline__ __device__ void update_indices()   {

    int mask_inds[2];
    uint32_t pred;
    pred = mask_[0] & (1u << (0 + 0));
    asm volatile (
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p,%1,0;\n"
        "  @p ld.global.b32 %0,[%2];\n"
        "}\n"
        : "=r"(mask_inds[0])
        : "r"(pred), "l"(params_.mask_argsort_ptr_)
    );
    pred = mask_[0] & (1u << (1 + 0));
    asm volatile (
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p,%1,0;\n"
        "  @p ld.global.b32 %0,[%2+32];\n"
        "}\n"
        : "=r"(mask_inds[1])
        : "r"(pred), "l"(params_.mask_argsort_ptr_)
    );
    #pragma unroll
    for (int s = 0; s < 2; ++s){
        #pragma unroll
        for (int ss = 0; ss < 1; ++ss){
            if (mask_[0] & (1u << (s * 1 + ss))){
                // 前向卷积: A 行偏移 = 输入voxel索引 * C(输入通道) * 2字节
                indices_[s * 1 + ss] =
                indice_ptr_[mask_inds[s * 1 + ss]] *
                    problem_.C * 2 ;
            }
        }
    }
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
  __forceinline__ __device__ void operator++()   {

  }
  __forceinline__ __device__ void increment_no_clear_mask()   {

  }
  __forceinline__ __device__ void clear_mask_if_batch_unbound()   {

  }
  __forceinline__ __device__ void operator+=(int num)   {

  }
  __forceinline__ __device__ void increment_k()   {

    pointer_ += params_.inc_c_next;
    reduce_channel_offset_ += params_.filter_c_delta;
    #pragma unroll
    for (int v = 0; v < 1; ++v){
        // 前向卷积: A 通道边界按 problem_.C 判断
        clear_mask_if_pred(reduce_channel_offset_ + v * 8 >= problem_.C, v);
    }
  }
  __forceinline__ __device__ void increment_filter()   {

    indice_ptr_ += problem_.N;
  }
  __forceinline__ __device__ void increment_filter(int num)   {

    indice_ptr_ += problem_.N * num;
  }
  __forceinline__ __device__ void reset_k()   {

    pointer_ += params_.inc_c_reset;
    mask_ = mask_reset_backup_;
    reduce_channel_offset_ = reduce_channel_offset_backup_;
  }
  __forceinline__ __device__ int get_indice_offset(int stride, int contig, int ss)  const {

    return indices_[stride * 1 + ss];
  }
  __forceinline__ __device__ const aligned_array<int4, 1, 16> * get(int indice_offset)  const {

    return reinterpret_cast<const aligned_array<int4, 1, 16> *>( pointer_ + indice_offset);
  }
  __forceinline__ __device__ void load_with_pointer_offset(std::array<half, 16>& frag, int32_t pointer_offset)   {

    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 16; ++i) frag[i] = half{};
    aligned_array<int4, 1, 16> *frag_ptr = reinterpret_cast<aligned_array<int4, 1, 16> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 2; ++s){
      #pragma unroll
      for (int c = 0; c < 1; ++c){
        #pragma unroll
        for (int ss = 0; ss < 1; ++ss){
          #pragma unroll
          for (int v = 0; v < 1; ++v){
            int mask_idx = s * 1 +
                c * 1 + ss;
            int idx = s * 1 +
                c * 1 + ss * 1 + v;
            auto indice_offset = get_indice_offset(s, c, ss);
            bool valid = bool(mask_[v] & (1u << mask_idx)) && (indice_offset >= 0);
            auto access_pointer = reinterpret_cast<const aligned_array<int4, 1, 16> *>(pointer_ + indice_offset +
                c * 64) + v;
            GlobalLoad::run(frag_ptr[idx], access_pointer, valid);
          }
        }
      }
    }
  }
  __forceinline__ __device__ const aligned_array<int4, 1, 16> * load_ptr_with_param(int s, int c, bool& valid_ref)   {

    int mask_idx = s * 1 +
        c * 1 + 0;
    auto indice_offset = get_indice_offset(s, c, 0);
    valid_ref = bool(mask_[0] & (1u << mask_idx)) && (indice_offset >= 0);
    auto access_pointer = reinterpret_cast<const aligned_array<int4, 1, 16> *>(pointer_ + indice_offset +
        c * 64) + 0;
    return access_pointer;
  }
  __forceinline__ __device__ void load(std::array<half, 16>& frag)   {
    load_with_pointer_offset(frag, 0);
  }
};
} // namespace conv
} // namespace cumm
