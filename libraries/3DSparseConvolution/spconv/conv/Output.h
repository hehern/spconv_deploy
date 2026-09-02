#pragma once
// 从 cumm 生成代码移植:
//   output/out_ns_frag/OutFragIterTensorOp.h
//   output/out_ns_warp/OutWarpTileIteratorTensorOp.h
//   output/out_ns_smem/OutSmemLoader.h (+ tmap/Out5DLinear.h)
//   output/out_ns_apply/ApplyOutputOp.h
//   output/Output.h
#include <cuda_fp16.h>
#include <vector_types.h>
#include <array>
#include <cstdint>
#include "ConvCommon.h"
#include "OutputSmemStorage.h"
#include "LinearCombination.h"
#include "OutIterator.h"
#include "ConstOutIterator.h"

namespace cumm {
namespace conv {

// ===== output/out_ns_frag/OutFragIterTensorOp =====
struct OutFragIterTensorOp {
  int index_;
  const aligned_array<int, 1, 4> * src_ptr_;
  __forceinline__ __device__  OutFragIterTensorOp(const void* src_ptr) : src_ptr_(reinterpret_cast<const aligned_array<int, 1, 4> *>(src_ptr)), index_(0)  {

  }
  __forceinline__ __device__ void load(std::array<half, 16> & frag, int32_t index_offset = 0)   {
    aligned_array<int, 1, 4> * frag_ptr = reinterpret_cast<aligned_array<int, 1, 4> *>(&frag);
    int index = index_ + index_offset;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        int accumulator_access_offset =
            index + n * 8 / 2;
        frag_ptr[n] = src_ptr_[accumulator_access_offset];
    }
  }
  __forceinline__ __device__ OutFragIterTensorOp& operator++()   {
    ++index_;
    return *this;
  }
};

// ===== output/out_ns_warp/OutWarpTileIteratorTensorOp =====
// RowMajor layout: offset(r, c) = r * 68 + c  (单位: 4B/int, 即 8 个 half)
struct OutWarpTileIteratorTensorOp {
  aligned_array<int, 1, 4> * pointer_;
  __forceinline__ __device__ static int64_t layout_offset(int32_t r, int32_t c)   {
    return int64_t(r) * 68 + c;
  }
  __forceinline__ __device__  OutWarpTileIteratorTensorOp(half * ptr, int warp_offset_m, int warp_offset_n, int lane_idx) : pointer_(reinterpret_cast<aligned_array<int, 1, 4> *>(ptr))  {

    int quad_id = (lane_idx / 4);
    int lane_in_quad = (lane_idx % 4);
    pointer_ += layout_offset(quad_id, lane_in_quad);
    add_warp_offset(warp_offset_m, warp_offset_n);
  }
  __forceinline__ __device__ void add_warp_offset(int warp_m, int warp_n)   {
    pointer_ += layout_offset(warp_m * 8, warp_n *
        64 / 2);
  }
  __forceinline__ __device__ void store_with_pointer_offset(std::array<half, 16> const & frag, int32_t pointer_offset)   {
    const aligned_array<int, 1, 4> * frag_ptr = reinterpret_cast<const aligned_array<int, 1, 4> *>(&frag);
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        pointer_[n * 4 + pointer_offset / 2] = frag_ptr[n];
    }
  }
  __forceinline__ __device__ void store(std::array<half, 16> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset / 2;
  }
};

// ===== output/out_ns_smem/tmap/Out5DLinear (smem loader 专用版本) =====
struct Out5DLinearSmem {
  __forceinline__ __host__ __device__ static std::array<int, 2> initial_offset(int thread_idx)   {
    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx %  32;
    // Compute warp location
    int cluster_idx = warp_idx / 4;
    int residual_cluster = warp_idx % 4;
    int group_idx = residual_cluster / 2;
    int residual_group = residual_cluster % 2;
    int row_idx = residual_group / 1;
    int col_idx = residual_group % 1;
    // Compute per-lane offset
    // in 1d warp, row offset always 0
    int lane_row_offset = lane_idx / 8;
    int lane_col_offset = lane_idx % 8;
    // in smem loader, x * 4 * 1 * 1 * 1 = x * 4
    int cluster_offset = cluster_idx * 2 * 1 *
                        8 * 1;
    int group_offset = group_idx * 8 * 1;
    // 0
    int row_offset = row_idx * 1 * 4; // 1d
    int column_offset =
        col_idx * 2 * 8 * 8;
    return {cluster_offset + group_offset + row_offset + lane_row_offset,
        (column_offset + lane_col_offset) * 8};
  }
};

// ===== output/out_ns_smem/OutSmemLoader =====
struct OutSmemLoader {
  half * pointer_;
  __forceinline__ __device__  OutSmemLoader(half * ptr, int thread_idx)   {
    auto thread_offset = Out5DLinearSmem::initial_offset(thread_idx);
    pointer_ = ptr + thread_offset[0] * 136 + thread_offset[1];
  }
  __forceinline__ __device__ void load_with_pointer_offset(std::array<half, 16> & frag, int32_t pointer_offset)   {
    #pragma unroll
    for (int cluster = 0; cluster < 1; ++cluster) {
        #pragma unroll
        for (int group = 0; group < 1; ++group) {
            #pragma unroll
            for (int row = 0; row < 1; ++row) {
                const half * cur_pointer =
                    pointer_ + row * 4 * 136 + group * 1 * 136 +
                    cluster * 1 * 136 + pointer_offset;
                int frag_row_idx =
                    (row + 1 * (group + 1 * cluster));
                aligned_array<int4, 1, 16> *frag_ptr = reinterpret_cast<aligned_array<int4, 1, 16> *>(&frag);
                aligned_array<int4, 1, 16> const *memory_pointer =
                    reinterpret_cast<aligned_array<int4, 1, 16> const *>(cur_pointer);
                #pragma unroll
                for (int column = 0; column < 2; ++column) {
                    int frag_idx = frag_row_idx * 2 + column;
                    #pragma unroll
                    for (int v = 0; v < 1; ++v) {
                        frag_ptr[frag_idx * 1 + v] =
                            memory_pointer[(column * 64 / 8) *
                                                1 +
                                            v];
                    }
                }
            }
        }
    }
  }
  __forceinline__ __device__ void load(std::array<half, 16> & frag)   {
    load_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ void add_pointer_offset(int pointer_offset)   {
    pointer_ += pointer_offset;
  }
};

// ===== output/out_ns_apply/ApplyOutputOp =====
struct ApplyOutputOp {
  __forceinline__ __device__ static void apply_output_operator(std::array<half, 16> & output_fragment, LinearCombination const & output_op, std::array<half, 16> const & aligned_accum_fragment, std::array<half, 16> const & source_fragment)   {

    using AccessType = std::array<half, 8>;
    AccessType *output_frag_ptr =
        reinterpret_cast<AccessType *>(&output_fragment);
    AccessType const *compute_frag_ptr =
        reinterpret_cast<AccessType const *>(&aligned_accum_fragment);
    AccessType const *source_frag_ptr =
        reinterpret_cast<AccessType const *>(&source_fragment);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
    }
  }
  __forceinline__ __device__ static void apply_output_operator_no_source(std::array<half, 16> & output_fragment, LinearCombination const & output_op, std::array<half, 16> const & aligned_accum_fragment)   {

    using AccessType = std::array<half, 8>;
    AccessType *output_frag_ptr =
        reinterpret_cast<AccessType *>(&output_fragment);
    AccessType const *compute_frag_ptr =
        reinterpret_cast<AccessType const *>(&aligned_accum_fragment);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
    }
  }
};

// ===== output/Output 主体 =====
using FragIter = OutFragIterTensorOp;
using OutWarpIter = OutWarpTileIteratorTensorOp;
using SmemLoader = OutSmemLoader;
using OutIter = OutIterator;
using ConstOutIter = ConstOutIterator;
using OutputStorage = out_smem_storage::OutputSmemStorage;
using OutputOp = LinearCombination;
using ApplyOp = ApplyOutputOp;

struct Output {
  OutWarpIter warp_iter;
  SmemLoader smem_loader;
  __forceinline__ __device__  Output(OutputStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter(smem_storage->smem.data_(), warp_idx_k * 2 + warp_m, warp_n, lane_idx), smem_loader(smem_storage->smem.data_(), thread_idx)  {

  }
  __forceinline__ __device__ void run(OutputOp const& output_op, std::array<half, 64> const& accumulators, OutIter& out_iter, ConstOutIter& source_iter)   {

    if (!output_op.is_source_needed()){
      return run_no_source(output_op, accumulators, out_iter);
    }
    std::array<half, 16> source_frag;
    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 16; ++i) source_frag[i] = half{};
    FragIter out_acc_iter(accumulators.data());
    #pragma unroll
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        std::array<half, 16> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2176);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        source_iter.load(source_frag);
        ++source_iter;
        std::array<half, 16> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2176);
        }
        std::array<half, 16> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
    }
  }
  __forceinline__ __device__ void run_no_source(OutputOp const& output_op, std::array<half, 64> const& accumulators, OutIter& out_iter)   {

    FragIter out_acc_iter(accumulators.data());
    #pragma unroll
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        std::array<half, 16> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2176);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        std::array<half, 16> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2176);
        }
        std::array<half, 16> out_frag;
        ApplyOp::apply_output_operator_no_source(out_frag, output_op, smem_frags[0]);
        out_iter.store(out_frag);
        ++out_iter;
      }
    }
  }
  __forceinline__ __device__ void run_self_reduce(OutputOp const& output_op, std::array<half, 64> const& accumulators, OutIter& out_iter)   {

    std::array<half, 16> source_frag;
    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 16; ++i) source_frag[i] = half{};
    FragIter out_acc_iter(accumulators.data());
    #pragma unroll
    for (int iter = 0; iter < 4; iter += 1){
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        std::array<half, 16> acc_frag;
        out_acc_iter.load(acc_frag);
        ++out_acc_iter;
        warp_iter.store(acc_frag);
        if (p < 1 - 1){
            warp_iter.add_pointer_offset(2176);
        }
      }
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < 1; ++p){
        out_iter.load(source_frag);
        std::array<half, 16> smem_frags[1];
        smem_loader.load(smem_frags[0]);
        if (p < 1 - 1){
            smem_loader.add_pointer_offset(2176);
        }
        std::array<half, 16> out_frag;
        ApplyOp::apply_output_operator(out_frag, output_op, smem_frags[0], source_frag);
        out_iter.store(out_frag);
        ++out_iter;
      }
    }
  }
};

} // namespace conv
} // namespace cumm
