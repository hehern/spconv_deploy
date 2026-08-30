#pragma once
#include <array>
#include <limits>
#include <cstdlib>    // std::abs
#include "ConvCommon.h"
namespace cumm {
namespace conv {

struct ConvProblem {
  int N;
  int C;
  int K;
  int kernel_volume;
  int split_k_slices;
  int groups;
  TV_HOST_DEVICE_INLINE  ConvProblem()   {
    
  }
  TV_HOST_DEVICE_INLINE  ConvProblem(int N, int C, int K, int kernel_volume, int split_k_slices = 1, int groups = 1) : N(N), C(C), K(K), kernel_volume(kernel_volume), split_k_slices(split_k_slices), groups(groups)  {
    
  }
  TV_HOST_DEVICE_INLINE std::array<int, 4> get_npq_shape()   {
    
    return {N};
  }
  TV_HOST_DEVICE_INLINE bool check_npq_not_overflow()   {
    
    auto shape = get_npq_shape();
    return (
    std::abs(int64_t(shape[0]) * int64_t(shape[1]) * int64_t(shape[2]) * int64_t(shape[3])) <= std::numeric_limits<int>::max()
    );
  }
  TV_HOST_DEVICE_INLINE static std::array<int, 3> calc_output_dims(std::array<int, 3> input_dims, std::array<int, 3> ksize, std::array<int, 3> padding, std::array<int, 3> stride, std::array<int, 3> dilation)   {
    
    std::array<int, 3> out;
    for (int i = 0; i < 3; ++i){
        out[i] = ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) / stride[i]) + 1;
    }
    return out;
  }
  TV_HOST_DEVICE_INLINE std::array<int, 3> implicit_gemm_mnk()   {
    
    return {N, K, C * kernel_volume};
  }
  TV_HOST_DEVICE_INLINE int implicit_gemm_k_iterations(int tile_shape_k)   {
    
    return kernel_volume * div_up(div_up(C, split_k_slices), tile_shape_k);
  }
  TV_HOST_DEVICE_INLINE std::array<int, 2> get_input_shape()   {
    
    return {N, C};
  }
  TV_HOST_DEVICE_INLINE std::array<int, 3> get_weight_shape()   {
    
    return {K, kernel_volume, C};
  }
  TV_HOST_DEVICE_INLINE std::array<int, 2> get_output_shape()   {
    
    return {N, K};
  }
  TV_HOST_DEVICE_INLINE static std::array<int, 3> conv_iwo_012_to_abc()   {
    
    return {0, 1, 2};
  }
  TV_HOST_DEVICE_INLINE static std::array<int, 3> gemm_abc_012_to_iwo()   {
    
    return {0, 1, 2};
  }

};

} // namespace conv
} // namespace cumm