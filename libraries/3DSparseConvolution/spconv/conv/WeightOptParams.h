#pragma once
#include "ConvProblem.h"
#include "lb/TensorGeneric.h"
namespace cumm {
namespace conv {

using Layout = lb::TensorGeneric;

struct WeightOptParams {
  Layout layout;
  int64_t inc_strided;
  int64_t inc_rs;
  int64_t inc_c;
  int filter_c_delta;
  int stride_rsc_bytes;
  int64_t inc_c_reset;
  __forceinline__ __host__ __device__  WeightOptParams(ConvProblem const& problem, Layout const& layout) : layout(layout)  {
    
    // int kernel_prod = problem.kernel_volume;
    filter_c_delta = 32 * problem.split_k_slices;
    // tnt(kForward): inc_strided 为 8 个 stride[0] (B 按 K 维 stride 访问, 每 8 个 K 一组)
    inc_strided = int64_t(layout.strides[0]) * 8;
    stride_rsc_bytes = layout.strides[0] * 16 / 8;
    // back to strided start, then inc c
    inc_c = filter_c_delta - inc_strided * int64_t(3);
    inc_rs = int64_t(layout.strides[1]);
    // inc_c_reset = -gemm_iters_k * filter_c_delta * 16 / 8;
    inc_rs = inc_rs * 16 / 8;
    inc_strided = inc_strided * 16 / 8;
    inc_c = inc_c * 16 / 8;
  }
  __forceinline__ __host__ __device__ void set_inc_reset_for_inc_k_first(int gemm_iters_k = -1)   {

    // tnt(kForward): gemm-K 沿输入通道 C 迭代, reset 只需 c 归零 (无 strides[0]);
    // ttt dgrad 版才是 -gemm_iters_k * filter_c_delta * layout.strides[0] * 16 / 8
    // (dgrad 的 gemm-K 沿输出通道 K 迭代, 需回退 K 行)。搬运时误用了 dgrad 公式,
    // 导致 reset_k 每次 filter 切换错误回退 filter_c_delta * strides[0] 字节,
    // 多个 filter 累计偏移达数百 KB, B 读取严重越界 (conv9 illegal address 根因)。
    inc_c_reset = -gemm_iters_k * filter_c_delta * 16 / 8;
  }
};

} // namespace conv
} // namespace cumm