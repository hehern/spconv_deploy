#pragma once
// 从 cumm 生成代码移植:
//   out_op/LinearCombination.h + out_op/unaryop/UnaryActivation.h
// tv::half_t -> half, tv::array -> std::array, tv::gemm::Activation -> cumm::conv::Activation
#include <cuda_fp16.h>
#include <array>
#include "ConvParams.h"

namespace cumm {
namespace conv {

// ===== out_op/unaryop/UnaryActivation =====
struct UnaryActivation {
  __forceinline__ __device__ std::array<half, 8> operator()(const std::array<half, 8> & src, Activation type, float alpha, float beta)   {

    std::array<half, 8> res;
    switch (type){
        case Activation::kNone:
            return src;
        case Activation::kReLU:{
            #pragma unroll
            for (int i = 0; i < 8; ++i){
                // CUDA 11.8 中 half 没有 half>=half 的精确定义, 需转换到 float 比较;
                // half 三元表达式公共类型计算同样歧义 (多个隐式转换), 统一改用 float + if/else
                if (__half2float(src[i]) >= 0.0f) {
                    res[i] = src[i];
                } else {
                    res[i] = half{0};
                }
            }
            return res;
        }
        case Activation::kLeakyReLU:{
            #pragma unroll
            for (int i = 0; i < 8; ++i){
                // 同上: half 之间 >= 比较与三元表达式均歧义, 转 float 处理
                float x = __half2float(src[i]);
                if (x >= 0.0f) {
                    res[i] = src[i];
                } else {
                    res[i] = __float2half(x * alpha);
                }
            }
            return res;
        }
        case Activation::kSigmoid:{
            #pragma unroll
            for (int i = 0; i < 8; ++i){
                float xx = __expf(-__half2float(src[i]));
                res[i] = __float2half(1.0f / (1.0f + xx));
            }
            return res;
        }
        default: return src;
    }
    return src;
  }
};

// ===== out_op/LinearCombination =====
struct LinearCombination {
  half alpha;
  half beta;
  half act_alpha;
  half act_beta;
  Activation act_type;
  __forceinline__ __device__  LinearCombination(half alpha = half{1}, half beta = half{0}, half act_alpha = half{0}, half act_beta = half{0}, Activation type = Activation::kNone) : alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(type)  {

  }
  __forceinline__ __device__ bool is_source_needed()  const {
    // half 之间 != 比较歧义, 转 float 比较
    return __half2float(beta) != 0.0f;
  }
  __forceinline__ __device__ void set_k_partition(int k_part, int k_part_count)   {

    if (k_part) {
        beta = half{1};
    }
  }
  __forceinline__ __device__ std::array<half, 8> operator()(std::array<half, 8> const& accumulator, std::array<half, 8> const& source)  const {

    // D = alpha * Accum + beta * Source
    std::array<half, 8> intermediate;
    #pragma unroll
    for (int i = 0; i < 8; ++i){
        intermediate[i] = __hfma(alpha, accumulator[i], __hmul(beta, source[i]));
    }
    UnaryOp op;
    intermediate = op(intermediate, act_type, __half2float(act_alpha), __half2float(act_beta));
    return intermediate;
  }
  __forceinline__ __device__ std::array<half, 8> operator()(std::array<half, 8> const& accumulator)  const {

    // D = alpha * Accum
    std::array<half, 8> intermediate;
    #pragma unroll
    for (int i = 0; i < 8; ++i){
        intermediate[i] = __hmul(alpha, accumulator[i]);
    }
    UnaryOp op;
    intermediate = op(intermediate, act_type, __half2float(act_alpha), __half2float(act_beta));
    return intermediate;
  }
  using UnaryOp = UnaryActivation;
};

} // namespace conv
} // namespace cumm
