#pragma once
// MmaMultiStage 及其全部子组件, 从 cumm 生成代码移植:
//   spconv/cumm/conv/main/Ampere_f16f16f16f16f16ttt_m64n128k32m32n64k32A1T1688_200_C311LLL_SK/mma/*
// tv::array -> std::array, tv::alignedarray -> aligned_array, TV_PRAGMA_UNROLL -> #pragma unroll
#include <cuda_fp16.h>
#include <vector_types.h>
#include <array>
#include <cstdint>
#include <cassert>
#include "ConvCommon.h"
#include "BlockMmaStorage.h"
#include "inpitera/tmap/PitchLinearWarpRaked.h"
#include "inpiterb/tmap/PitchLinearWarpRaked.h"
#include "ForwardDgradSparseIOIterator.h"
#include "WeightIteratorDP4A.h"

namespace cumm {
namespace conv {

// ===== mma_ns_miter/MaskIGemmIterator =====
struct MaskIGemmIterator {
  int filter_idx;
  const int& gemm_k_iterations;
  const int& RS;
  const uint32_t& mask;
  bool end;
  __forceinline__ __device__  MaskIGemmIterator(const int& gemm_k_iterations, const int& RS, const uint32_t& mask) : filter_idx(0), gemm_k_iterations(gemm_k_iterations), RS(RS), mask(mask), end(false)  {

  }
  __forceinline__ __device__ void operator++()   {

    if (++filter_idx < RS){
        return;
    }
    end = true;
  }
  __forceinline__ __device__ bool valid()  const {

    return mask & (1u << filter_idx);
  }
};

// ===== mma_miterd/MaskIGemmIteratorMaskLoaderDynamic =====
struct MaskIGemmIteratorMaskLoaderDynamic {
  int current_mask;
  const int& mask_int_count;
  int mask_load_idx;
  int RS_pos;
  int RS_offset;
  const uint32_t & mask_filter;
  const int& tile_offset_m;
  const uint32_t* const& mask_ptr;
  uint32_t* const& mask_out_ptr;
  const int& RS;
  const bool& reverse;
  const int& gemm_k_iteration;
  const int& lane_idx;
  const int& problem_m;
  bool end;
  int mask_or_sum;
  __forceinline__ __device__  MaskIGemmIteratorMaskLoaderDynamic(const uint32_t* const& mask_ptr, uint32_t* const& mask_out_ptr, const int& mask_int_count, const int& tile_offset_m, const int& gemm_k_iteration, const int& RS, const uint32_t& mask_filter, const bool& reverse, const int& lane_idx, const int& problem_m) : mask_int_count(mask_int_count), mask_ptr(mask_ptr), mask_out_ptr(mask_out_ptr), tile_offset_m(tile_offset_m), RS(RS), gemm_k_iteration(gemm_k_iteration), end(false), reverse(reverse), mask_filter(mask_filter), lane_idx(lane_idx), problem_m(problem_m)  {

    mask_or_sum = 0;
    int used_rs = div_up(RS, 32);
    if (reverse){
      for (mask_load_idx = 0; mask_load_idx < used_rs - 1; ++mask_load_idx){
          load_mask(true);
          mask_or_sum |= __brev(current_mask);
      }
    }
    else{
      for (mask_load_idx = 1; mask_load_idx < used_rs; ++mask_load_idx){
          load_mask(true);
          mask_or_sum |= current_mask;
      }
    }
    init_mask_iter(true);
    mask_or_sum |= current_mask;
  }
  __device__ inline void load_mask(const bool& save = false)   {

    current_mask = 0;
    std::array<uint32_t, 2> masks;
    // std::array::fill 是 __host__ 函数, 设备端用循环初始化
    #pragma unroll
    for (int i = 0; i < 2; ++i) masks[i] = 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i){
        if (tile_offset_m * 64 + i * 32 + lane_idx < problem_m){
            masks[i] = mask_ptr[mask_int_count * (tile_offset_m * 64 + i * 32 + lane_idx) + mask_load_idx];
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i){
        current_mask |= masks[i];
    }
    // perform a warp reduce to get block mask
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        current_mask |= __shfl_xor_sync(0xffffffff, current_mask, mask, 32);
    }
    current_mask &= mask_filter;
  }
  __device__ inline void init_mask_iter(const bool& save = false)   {

    if (reverse){
        int used_mask = div_up(RS, 32);
        mask_load_idx = used_mask - 1;
        RS_pos = 0;
        RS_offset = used_mask * 32 - RS;
        load_mask(save);
        current_mask = __brev(current_mask);
        return;
    }
    mask_load_idx = 0;
    RS_pos = RS_offset = 0;
    load_mask(save);
  }
  __device__ inline void operator++()   {

    if (++RS_pos >= RS){
      end = true;
      return;
    }
    if ((RS_pos + RS_offset) % 32 == 0){
      if (reverse){
        --mask_load_idx;
      }
      else{
        ++mask_load_idx;
      }
      load_mask();
      if (reverse){
        current_mask = __brev(current_mask);
      }
    }
  }
  __device__ inline bool valid()   {

    return !!(
            current_mask & (1u << ((RS_pos + RS_offset) % 32))
            );
  }
  __device__ inline bool empty()   {

    return !mask_or_sum;
  }
};

// ===== mma_ns_wa/layout/MyTensorOpLayout (crosswise) =====
namespace mma_ns_wa {
namespace layout {
struct MyTensorOpLayout {
  __forceinline__ __host__ __device__ constexpr  MyTensorOpLayout()   {

  }
  __forceinline__ __host__ __device__ constexpr static MyTensorOpLayout from_shape(const std::array<int, 2> & shape)   {
    return MyTensorOpLayout();
  }
  __forceinline__ __host__ __device__ constexpr int64_t operator()(int32_t s, int32_t ec)  const {

    int vc = ec / 8;
    int interleaved_s = s / 2;
    int idx_in_interleave_s = s % 2;
    int sw_idx_c = vc / 4;
    int idx_in_sw_c = vc % 4 + idx_in_interleave_s * 4;
    int idx_in_sw_s = interleaved_s % 4;
    int subsw_idx_s = idx_in_sw_s / 4;
    int subsw_idx_c = idx_in_sw_c / 4;
    int idx_in_subsw_s = idx_in_sw_s % 4;
    int idx_in_subsw_c = idx_in_sw_c % 4;
    int permuted_subsw_idx_c = subsw_idx_c;
    if (1 > 1){
        permuted_subsw_idx_c = subsw_idx_c ^ (subsw_idx_s % 2);
    }
    int premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % 4);
    int final_c = sw_idx_c * 8 + permuted_subsw_idx_c * 4 + premuted_idx_in_subsw_c;
    int final_ec = final_c * 8 + ec % 8;
    int final_s = interleaved_s * 128;
    return final_ec + final_s;
  }
  template <int LdmCountStride, int LdmCountContig>
  __forceinline__ __host__ __device__ constexpr static int64_t get_ldm_initial_offset(int32_t lane_idx, int32_t permute_m_pointer_idx, bool transpose)   {

    int stride = -1;
    int contig_vec = -1;
    if (LdmCountContig == 1){
        stride = lane_idx >> 1;
        contig_vec = ((lane_idx >> 1) & 0b11) ^ ((lane_idx & 1) << 2) ^ permute_m_pointer_idx;
    } else if (LdmCountContig == 2 && LdmCountStride == 2){
        if (transpose){
            int _00112233 = ((lane_idx >> 1) & 0b11);
            stride = _00112233 + (lane_idx >> 4 << 2);
            contig_vec = (_00112233 + ((lane_idx & 1) << 2)) ^ ((lane_idx >> 3) & 1) ^ (permute_m_pointer_idx << 1);
        }else{
            stride = (lane_idx & 0b1111) >> 1;
            contig_vec = ((((lane_idx >> 1) & 0b11) + ((lane_idx & 1) << 2)) ^ (lane_idx >> 4)) ^ (permute_m_pointer_idx << 1);
        }
    }else if (LdmCountContig == 2 && LdmCountStride == 1){
        stride = (lane_idx & 0b111) >> 1;
        contig_vec = (((lane_idx >> 1) & 0b11) ^ ((lane_idx & 1) << 2)) ^ (lane_idx >> 3) ^ (permute_m_pointer_idx << 1);
    } else{
        stride = (lane_idx & 0b111) >> 1;
        contig_vec = (((lane_idx >> 1) & 0b11) + ((lane_idx & 1) << 2)) ^ (lane_idx >> 3);
    }
    return stride * 128 + contig_vec * 8;
  }
};
} // namespace layout
} // namespace mma_ns_wa

// ===== mma_ns_wb/layout/MyTensorOpLayout (congruous) =====
namespace mma_ns_wb {
namespace layout {
struct MyTensorOpLayout {
  __forceinline__ __host__ __device__ constexpr  MyTensorOpLayout()   {

  }
  __forceinline__ __host__ __device__ constexpr static MyTensorOpLayout from_shape(const std::array<int, 2> & shape)   {
    return MyTensorOpLayout();
  }
  __forceinline__ __host__ __device__ constexpr int64_t operator()(int32_t s, int32_t ec)  const {

    int vc = ec / 8;
    int interleaved_s = s / 1;
    int idx_in_interleave_s = s % 1;
    int sw_idx_c = vc / 8;
    int idx_in_sw_c = vc % 8 + idx_in_interleave_s * 8;
    int idx_in_sw_s = interleaved_s % 8;
    int subsw_idx_s = idx_in_sw_s / 4;
    int subsw_idx_c = idx_in_sw_c / 4;
    int idx_in_subsw_s = idx_in_sw_s % 4;
    int idx_in_subsw_c = idx_in_sw_c % 4;
    int permuted_subsw_idx_c = subsw_idx_c;
    if (2 > 1){
        permuted_subsw_idx_c = subsw_idx_c ^ (subsw_idx_s % 2);
    }
    int premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % 4);
    int final_c = sw_idx_c * 8 + permuted_subsw_idx_c * 4 + premuted_idx_in_subsw_c;
    int final_ec = final_c * 8 + ec % 8;
    int final_s = interleaved_s * 128;
    return final_ec + final_s;
  }
  template <int LdmCountStride, int LdmCountContig>
  __forceinline__ __host__ __device__ constexpr static int64_t get_ldm_initial_offset(int32_t lane_idx, int32_t permute_m_pointer_idx, bool transpose)   {

    int stride = -1;
    int contig_vec = -1;
    if (LdmCountContig == 1){
        stride = lane_idx;
        contig_vec = (lane_idx & 0b111) ^ permute_m_pointer_idx;
    } else if (LdmCountContig == 2 && LdmCountStride == 2){
        if (transpose){
            int _01234567 = (lane_idx & 0b111);
            stride = _01234567 + (lane_idx >> 4 << 3);
            contig_vec = _01234567 ^ (((lane_idx >> 3) & 1) + (permute_m_pointer_idx << 1));
        }else{
            int _01234567 = (lane_idx & 0b111);
            stride = lane_idx & 0b1111;
            contig_vec = _01234567 ^ ((lane_idx >> 4) + (permute_m_pointer_idx << 1));
        }
    }else if (LdmCountContig == 2 && LdmCountStride == 1){
        stride = lane_idx & 0b111;
        contig_vec = stride ^ ((lane_idx >> 3) ^ (permute_m_pointer_idx << 1));
    }else {
        stride = lane_idx & 0b111;
        contig_vec = stride ^ ((lane_idx >> 3) + (permute_m_pointer_idx << 2));
    }
    return stride * 128 + contig_vec * 8;
  }
};
} // namespace layout
} // namespace mma_ns_wb

// ===== ldmatrix (x4 / x4.trans) =====
struct LdMatrixX4 {
  __forceinline__ __device__ static void run(std::array<unsigned, 4> & D, void const* ptr)   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
      unsigned addr = get_smem_pointer(ptr);
      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #else
      assert(0);
    #endif
  }
};

struct LdMatrixX4Trans {
  __forceinline__ __device__ static void run(std::array<unsigned, 4> & D, void const* ptr)   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
      unsigned addr = get_smem_pointer(ptr);
      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #else
      assert(0);
    #endif
  }
};

// ===== mma_ns_wa/WarpIteratorCrosswise =====
class WarpIteratorCrosswise {
 public:
  using TensorOpLayout = mma_ns_wa::layout::MyTensorOpLayout;
  using LdMatrix = LdMatrixX4;
  const std::array<half, 8> * pointer_;
  int32_t byte_offset_;
  int wmma_k_index_;
  __forceinline__ __device__  WarpIteratorCrosswise(half * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx) : pointer_(reinterpret_cast<const std::array<half, 8> *>(ptr)), wmma_k_index_(0), byte_offset_(0)  {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
        lane_idx = lane_idx % (4 * 8);
    #endif
    int offset_e = TensorOpLayout::get_ldm_initial_offset<4, 1>(
        lane_idx, 0, false);
    byte_offset_ = offset_e * 16 / 8;
    add_tile_offset(4 * warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_tile_offset(int warp_idx_k, int warp_idx_mn)   {
    int mn_offset = warp_idx_mn;
    int k_offset = warp_idx_k;
    int sw_part_idx = k_offset / 4;
    int idx_in_sw_part = k_offset % 4;
    byte_offset_ ^= (idx_in_sw_part * 16);
    pointer_ +=
        mn_offset * 256 +
        sw_part_idx * 8;
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(num_tile, 0);
  }
  __forceinline__ __device__ WarpIteratorCrosswise & operator++()   {

    if (((wmma_k_index_ & 1) & 1) == 0){
        // bit 0 advance
        byte_offset_ ^= 0b1 * 16;
    }
    else if ((wmma_k_index_ & 1) == 0b1){
        // bit 1 advance
        byte_offset_ ^= 0b11 * 16;
    }
    else if ((wmma_k_index_ & 1) == 0b11){
        // bit 2 advance
        byte_offset_ ^= 0b111 * 16;
    }
    wmma_k_index_++;
    if (wmma_k_index_ == 4) {
        wmma_k_index_ = 0;
        // k group increment
        add_tile_offset(4, 0);
    }
    return *this;
  }
  __forceinline__ __device__ void load_with_byte_offset(std::array<half, 8>& frag, int32_t byte_offset)   {
    std::array<unsigned, 4> *fetch_ptr =
        reinterpret_cast<std::array<unsigned, 4> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 1; ++s) {
        #pragma unroll
        for (int c = 0; c < 1; ++c) {
            int access_idx = c + s * 1;
            const std::array<half, 8> * source_ptr =
                pointer_ + 1 * c +
                8 * 4 * s *
                8;
            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;
            LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
        }
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(std::array<half, 8>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(half));
  }
  __forceinline__ __device__ void load(std::array<half, 8>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {
    wmma_k_index_ = wmma_k % (4);
  }
};

// ===== mma_ns_wb/WarpIteratorCongruous =====
class WarpIteratorCongruous {
 public:
  using TensorOpLayout = mma_ns_wb::layout::MyTensorOpLayout;
  using LdMatrix = LdMatrixX4Trans;
  int wmma_k_index_;
  const std::array<half, 8> * pointer_[2];
  int32_t byte_offset_;
  __forceinline__ __device__  WarpIteratorCongruous(half * ptr, int warp_idx_k, int warp_idx_mn, int lane_idx) : wmma_k_index_(0), byte_offset_(0)  {

    lane_idx %= 32;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int offset = TensorOpLayout::get_ldm_initial_offset<1, 4>(
            lane_idx, i, true);
        pointer_[i] = reinterpret_cast<const std::array<half, 8> * >(ptr + offset);
    }
    add_tile_offset(4 * warp_idx_k, warp_idx_mn);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(half);
  }
  __forceinline__ __device__ void add_tile_offset(int warp_idx_k, int warp_idx_mn, bool force_update = false)   {

        constexpr int kContigEqual = 64;
    int mn_offset = warp_idx_mn;
    int k_offset = warp_idx_k;
    if (64 < kContigEqual || force_update) {
      constexpr int kwarp_per_crosswise = 1;
      int warp_offset = warp_idx_mn & (kwarp_per_crosswise - 1);
      mn_offset = warp_idx_mn ^ warp_offset;            // kwarp_per_crosswise is 2^a
      warp_offset *= 2 / kwarp_per_crosswise;
      if (warp_offset || force_update) {
    const std::array<half, 8> * buffer[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      buffer[i] = pointer_[(i + warp_offset) % 2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      pointer_[i] = buffer[i];
      }
    }
    int offset = (k_offset * 1024 +
                mn_offset * 64);
    add_pointer_offset(offset);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(num_tile, 0);
  }
  __forceinline__ __device__ WarpIteratorCongruous & operator++()   {
    add_tile_offset(1, 0); // strided, contig
    return *this;
  }
  __forceinline__ __device__ void load_with_byte_offset(std::array<half, 16>& frag, int32_t byte_offset)   {
    std::array<unsigned, 4> *fetch_ptr =
    reinterpret_cast<std::array<unsigned, 4> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 1; ++s) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int access_idx = c + s * 2;
            const std::array<half, 8> * source_ptr =
                pointer_[c % 2] +
                8 * (c / 2) +
                1 * s * 16;
            char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;
            LdMatrix::run(fetch_ptr[access_idx], source_byte_ptr);
        }
    }
  }
  __forceinline__ __device__ void load_with_pointer_offset(std::array<half, 16>& frag, int32_t pointer_offset)   {
    load_with_byte_offset(frag, pointer_offset * sizeof(half));
  }
  __forceinline__ __device__ void load(std::array<half, 16>& frag)   {
    load_with_byte_offset(frag, 0);
  }
  __forceinline__ __device__ void set_kgroup_index(int wmma_k)   {

  }
};

// ===== mma_ns_sa/SmemTileIterator (A operand) =====
class SmemTileIteratorA {
 public:
  using ThreadMap = inpitera::tmap::PitchLinearWarpRaked;
  using Layout = mma_ns_wa::layout::MyTensorOpLayout;
  std::array<half, 8> * pointer_[1];
  int32_t byte_offset_;
  __forceinline__ __device__  SmemTileIteratorA(int stride, half * ptr, int thread_id) : byte_offset_(0)  {

    auto thread_offset_base = ThreadMap::initial_offset(thread_id);
    auto layout = Layout();
    #pragma unroll
    for (int i = 0; i < 1; ++i) {
        pointer_[i] = reinterpret_cast<std::array<half, 8> *>(
            ptr + layout(thread_offset_base[0] + i * 8,
                        thread_offset_base[1]));
    }
  }
  __forceinline__ __device__ std::array<half, 8> * get(int s, int c)  const {
    std::array<half, 8> * access_ptr = pointer_[s & 0];
    int external_stride_idx = (s & ~0);
    int access_offset = (external_stride_idx * 4 *
                     16 + c * 8);
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<std::array<half, 8> *>(access_byte_ptr + byte_offset_);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(half);
  }
  __forceinline__ __device__ void add_tile_offset(int s, int c)   {
    add_pointer_offset(c * 64 +
        s * 4096);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(0, num_tile);
  }
  __forceinline__ __device__ void store_with_pointer_offset(std::array<half, 16> const& frag, int32_t pointer_offset)   {
    store_with_byte_offset(frag, pointer_offset * 16 / 8);
  }
  __forceinline__ __device__ void store_with_byte_offset(std::array<half, 16> const& frag, int32_t byte_offset)   {

    const std::array<half, 8> * frag_ptr = reinterpret_cast<const std::array<half, 8> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int c = 0; c < 1; ++c) {
            int access_idx = c + s * 1;
            char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
            std::array<half, 8> * access_ptr = reinterpret_cast<std::array<half, 8> *>(byte_ptr);
            *access_ptr = frag_ptr[access_idx];
        }
    }
  }
  __forceinline__ __device__ std::array<half, 8> * store_ptr_with_param(int s, int c, bool& valid_ref)   {

    return reinterpret_cast<std::array<half, 8> *>(get(s, c));
  }
  __forceinline__ __device__ void store(std::array<half, 16> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ SmemTileIteratorA & operator++()   {
    add_tile_offset(0, 1);
    return *this;
  }
};

// ===== mma_ns_sb/SmemTileIterator (B operand) =====
class SmemTileIteratorB {
 public:
  using ThreadMap = inpiterb::tmap::PitchLinearWarpRaked;
  using Layout = mma_ns_wb::layout::MyTensorOpLayout;
  std::array<half, 8> * pointer_[2];
  int32_t byte_offset_;
  __forceinline__ __device__  SmemTileIteratorB(int stride, half * ptr, int thread_id) : byte_offset_(0)  {

    auto thread_offset_base = ThreadMap::initial_offset(thread_id);
    auto layout = Layout();
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        pointer_[i] = reinterpret_cast<std::array<half, 8> *>(
            ptr + layout(thread_offset_base[0] + i * 4,
                        thread_offset_base[1]));
    }
  }
  __forceinline__ __device__ std::array<half, 8> * get(int s, int c)  const {
    std::array<half, 8> * access_ptr = pointer_[s & 1];
    int external_stride_idx = (s & ~1);
    int access_offset = (external_stride_idx * 4 *
                     16 + c * 8);
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<std::array<half, 8> *>(access_byte_ptr + byte_offset_);
  }
  __forceinline__ __device__ void add_pointer_offset(int64_t offset)   {
    byte_offset_ += offset * sizeof(half);
  }
  __forceinline__ __device__ void add_tile_offset(int s, int c)   {
    add_pointer_offset(c * 128 +
        s * 4096);
  }
  __forceinline__ __device__ void tile_increment(int num_tile)   {
    add_tile_offset(num_tile, 0);
  }
  __forceinline__ __device__ void store_with_pointer_offset(std::array<half, 32> const& frag, int32_t pointer_offset)   {
    store_with_byte_offset(frag, pointer_offset * 16 / 8);
  }
  __forceinline__ __device__ void store_with_byte_offset(std::array<half, 32> const& frag, int32_t byte_offset)   {

    const std::array<half, 8> * frag_ptr = reinterpret_cast<const std::array<half, 8> *>(&frag);
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int access_idx = c + s * 2;
            char *byte_ptr = reinterpret_cast<char *>(get(s, c)) + byte_offset;
            std::array<half, 8> * access_ptr = reinterpret_cast<std::array<half, 8> *>(byte_ptr);
            *access_ptr = frag_ptr[access_idx];
        }
    }
  }
  __forceinline__ __device__ std::array<half, 8> * store_ptr_with_param(int s, int c, bool& valid_ref)   {

    return reinterpret_cast<std::array<half, 8> *>(get(s, c));
  }
  __forceinline__ __device__ void store(std::array<half, 32> const& frag)   {
    store_with_pointer_offset(frag, 0);
  }
  __forceinline__ __device__ SmemTileIteratorB & operator++()   {
    add_tile_offset(1, 0);
    return *this;
  }
};

// ===== CpAsyncGroup =====
struct CpAsyncGroup {
  __forceinline__ __device__ static void make_fence()   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      asm volatile("cp.async.commit_group;\n" ::);
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void wait_final_group()   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void wait_all()   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      asm volatile("cp.async.wait_all;\n" ::);
    #else
      assert(0);
    #endif
  }
};

// ===== cp_async_copy/CpAsyncCopy (16B, cg) =====
struct CpAsyncCopy16B {
  __forceinline__ __device__ static void copy(void* dest_smem, const void* src_global, bool pred_guard = true)   {

    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
                          unsigned smem_addr = get_smem_pointer(dest_smem);
                          asm volatile(
                              "{\n"
                              "  .reg .pred p;\n"
                              "  setp.ne.b32 p, %0, 0;\n"
                              "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
                              "}\n" ::"r"((int)pred_guard), "r"(smem_addr), "l"(src_global), "n"(16));
    #else
      assert(0);
    #endif
  }
  __forceinline__ __device__ static void copy_zfill(void* dest_smem, const void* src_global, bool pred_guard = true)   {


    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      unsigned smem_addr = get_smem_pointer(dest_smem);
      unsigned real_size = (pred_guard ? 16 : 0);
                          asm volatile(
                              "cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
                               ::"r"(smem_addr), "l"(src_global), "n"(16), "r"(real_size));
    #else
      assert(0);
    #endif
  }
};

// ===== async_cp_iter_*/AsyncCopyIteration (10 个变体) =====
struct AsyncCopyIterGlobalA {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIterGlobalB {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 1, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 1, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 1, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 1, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter0A {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    ///// nothing to do here /////
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    ///// nothing to do here /////
  }
};

struct AsyncCopyIter0B {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter1A {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter1B {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 1, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(0, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(0, 1, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter2A {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {
    ///// nothing to do here /////
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {
    ///// nothing to do here /////
  }
};

struct AsyncCopyIter2B {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter3A {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 0, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 0, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

struct AsyncCopyIter3B {
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 1, valid);
    CpAsyncCopy16B::copy(dest_ptr, src_ptr, valid);
  }
  template <typename InputIter, typename SmemIter>
  __forceinline__ __device__ static void do_copy_zfill(InputIter& input_iter, SmemIter& smem_iter)   {

    bool valid;
    const void* src_ptr;
    void* dest_ptr;
    valid = true;
    src_ptr = input_iter.load_ptr_with_param(1, 1, valid);
    dest_ptr = smem_iter.store_ptr_with_param(1, 1, valid);
    CpAsyncCopy16B::copy_zfill(dest_ptr, src_ptr, valid);
  }
};

// ===== mma_ns_wmma/WarpMmaTuring (m16n8k8 mma.sync) =====
struct WarpMmaTuring {
  __forceinline__ __device__ void operator()(std::array<half, 64>& D, std::array<half, 8> const & A, std::array<half, 16> const & B, std::array<half, 64> const & C)   {

    D = C;
    std::array<half, 4> const *ptr_A = reinterpret_cast<std::array<half, 4> const *>(&A);
    std::array<half, 2> const *ptr_B = reinterpret_cast<std::array<half, 2> const *>(&B);
    std::array<half, 4> *ptr_D = reinterpret_cast<std::array<half, 4> *>(&D);
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
      #pragma unroll
      for (int n = 0; n < 8; ++n){
        #pragma unroll
        for (int m = 0; m < 2; ++m){
          int m_serpentine = ((n % 2) ? (2 - 1 - m) : m);
          mma_m16n8k8(ptr_D[m_serpentine + n * 2],
              ptr_A[m_serpentine],
              ptr_B[n],
              ptr_D[m_serpentine + n * 2]);
        }
      }
    #elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
      #pragma unroll
      for (int m = 0; m < 2; ++m){
        #pragma unroll
        for (int n = 0; n < 8; ++n){
          int n_serpentine = ((m % 2) ? (8 - 1 - n) : n);
          mma_m16n8k8(ptr_D[m + n_serpentine * 2],
              ptr_A[m],
              ptr_B[n_serpentine],
              ptr_D[m + n_serpentine * 2]);
        }
      }
    #endif
  }
  __forceinline__ __device__ void mma_m16n8k8(std::array<half, 4>& d, std::array<half, 4> const & a, std::array<half, 2> const & b, std::array<half, 4> const & c)   {
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
    unsigned const *A = reinterpret_cast<unsigned const *>(&a);
    unsigned const & B = reinterpret_cast<unsigned const &>(b);
    unsigned const *C = reinterpret_cast<unsigned const *>(&c);
    unsigned *D = reinterpret_cast<unsigned *>(&d);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
        : "=r"(D[0]), "=r"(D[1])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]));
    #endif
  }
};

// ===== MmaMultiStage 主体 =====
using InputIteratorA = ForwardDgradSparseIOIterator;
using InputIteratorB = WeightIteratorDP4A;
using WarpMma = WarpMmaTuring;
using GemmStorage = gemm_smem_storage::BlockMmaStorage;

struct MmaMultiStage {
  WarpIteratorCrosswise warp_iter_A;
  WarpIteratorCongruous warp_iter_B;
  SmemTileIteratorA smem_iter_A;
  SmemTileIteratorB smem_iter_B;
  __forceinline__ __device__  MmaMultiStage(GemmStorage* smem_storage, int thread_idx, int warp_idx_k, int warp_m, int warp_n, int lane_idx) : warp_iter_A(smem_storage->smem_A.data_(), warp_idx_k, warp_m, lane_idx), warp_iter_B(smem_storage->smem_B.data_(), warp_idx_k, warp_n, lane_idx), smem_iter_A(64, smem_storage->smem_A.data_(), thread_idx), smem_iter_B(128, smem_storage->smem_B.data_(), thread_idx)  {

  }
  __forceinline__ __device__ void copy_tiles_and_advance(InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, const int & group_idx)   {

                        if(group_idx == 0){
                          AsyncCopyIter0A::do_copy_zfill(input_iter_A, smem_iter_A);
                          AsyncCopyIter0B::do_copy_zfill(input_iter_B, smem_iter_B);
                          return;
                        }
                        if(group_idx == 1){
                            AsyncCopyIter1A::do_copy_zfill(input_iter_A, smem_iter_A);
                            AsyncCopyIter1B::do_copy_zfill(input_iter_B, smem_iter_B);
                            return;
                        }
                        if(group_idx == 2){
                            AsyncCopyIter2A::do_copy_zfill(input_iter_A, smem_iter_A);
                            AsyncCopyIter2B::do_copy_zfill(input_iter_B, smem_iter_B);
                            return;
                        }
                        if(group_idx == 3){
                            AsyncCopyIter3A::do_copy_zfill(input_iter_A, smem_iter_A);
                            AsyncCopyIter3B::do_copy_zfill(input_iter_B, smem_iter_B);
                            return;
                        }
  }
  __forceinline__ __device__ void operator()(const int& gemm_k_iterations, std::array<half, 64>& accumulators, InputIteratorA & input_iter_A, InputIteratorB & input_iter_B, std::array<half, 64> const& src_accumulators, uint32_t mask, const int& RS)   {

    accumulators = src_accumulators;
    std::array<half, 8> warp_frag_A[2];
    std::array<half, 16> warp_frag_B[2];
    WarpMma warp_mma;
    int smem_write_stage_idx = 1;
    int smem_read_stage_idx = 0;
    MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
    int local_gemm_k_iterations = gemm_k_iterations;
    while(!mask_iter.valid()){
        ++mask_iter;
        input_iter_A.increment_filter();
        input_iter_B.increment_filter();
    }
    input_iter_A.update_indices();
    #pragma unroll
    for (int stage=0; stage < 1; ++stage){
        AsyncCopyIterGlobalA::do_copy_zfill(input_iter_A, smem_iter_A);
        AsyncCopyIterGlobalB::do_copy_zfill(input_iter_B, smem_iter_B);
        CpAsyncGroup::make_fence();
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        ++smem_iter_A;
        ++smem_iter_B;
        --local_gemm_k_iterations;
        if (!mask_iter.end && local_gemm_k_iterations == 0){
            ++mask_iter;
            input_iter_A.reset_k();
            input_iter_B.reset_k();
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
            while (!mask_iter.valid() && !mask_iter.end){
                ++mask_iter;
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();
            }
            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
            input_iter_A.update_indices();
            local_gemm_k_iterations = gemm_k_iterations;
        }
    }
    CpAsyncGroup::wait_final_group();
    __syncthreads();
    warp_iter_A.set_kgroup_index(0);
    warp_iter_B.set_kgroup_index(0);
    warp_iter_A.load(warp_frag_A[0]);
    warp_iter_B.load(warp_frag_B[0]);
    ++warp_iter_A;
    ++warp_iter_B;
    while (local_gemm_k_iterations != -1){
        #pragma unroll
        for (int warp_mma_k = 0; warp_mma_k < 4; ++warp_mma_k){
            warp_iter_A.set_kgroup_index((warp_mma_k + 1) % 4);
            warp_iter_B.set_kgroup_index((warp_mma_k + 1) % 4);
            warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
            warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
            ++warp_iter_A;
            ++warp_iter_B;
            warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                warp_frag_B[warp_mma_k % 2], accumulators);
            if (warp_mma_k < 3)
                copy_tiles_and_advance(input_iter_A, input_iter_B, warp_mma_k);
            if (warp_mma_k + 2 == 4){
                copy_tiles_and_advance(input_iter_A, input_iter_B, 3);
                CpAsyncGroup::make_fence();
                // do chores before wait
                ++smem_iter_A;
                ++smem_iter_B;
                input_iter_A.increment_k();
                input_iter_B.increment_k();
                --local_gemm_k_iterations;
                if (!mask_iter.end && local_gemm_k_iterations == 0){
                    ++mask_iter;
                    input_iter_A.reset_k();
                    input_iter_B.reset_k();
                    input_iter_A.increment_filter();
                    input_iter_B.increment_filter();
                    while (!mask_iter.valid() && !mask_iter.end){
                        ++mask_iter;
                        input_iter_A.increment_filter();
                        input_iter_B.increment_filter();
                    }
                    input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                    input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                    input_iter_A.update_indices();
                    local_gemm_k_iterations = gemm_k_iterations;
                }
                if (smem_write_stage_idx == 1) {
                    smem_iter_A.tile_increment(-2);
                    smem_iter_B.tile_increment(-2);
                    smem_write_stage_idx = 0;
                } else
                    ++smem_write_stage_idx;
                if (smem_read_stage_idx == 1) {
                    warp_iter_A.tile_increment(-2 *
                                            4);
                    warp_iter_B.tile_increment(-2 *
                                            4);
                    smem_read_stage_idx = 0;
                } else
                    ++smem_read_stage_idx;
                // finish chores
                CpAsyncGroup::wait_final_group();
                __syncthreads();
            }
        }
    }
  }
};

} // namespace conv
} // namespace cumm
