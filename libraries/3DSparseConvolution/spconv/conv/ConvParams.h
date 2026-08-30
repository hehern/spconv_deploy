#pragma once
#include <cuda_fp16.h>
#include <vector_types.h>
#include <cassert>
#include "ConvProblem.h"
#include "ConvUtils.h"
#include "SparseParams.h"
#include "WeightOptParams.h"
#include "OutIteratorParams.h"
// #include "la/TensorGeneric.h"
#include "lb/TensorGeneric.h"
// #include "lc/TensorGeneric.h"
namespace cumm {
namespace conv {

using LayoutB = lb::TensorGeneric;

enum class Activation {
  // we only support three activations here.
  kNone = 0,
  kReLU = 1,
  kSigmoid = 2,
  kLeakyReLU = 3,
  // kELU = 5,
  // kSeLU = 6,
  // kSoftsign = 7,
  // kSoftplus = 8,
  // kClip = 9,
  // kHardSigmoid = 10,
  // kScaledTanh = 11,
  // kThresholdedReLU = 12

};

struct ConvParams {
  ConvProblem problem;
  int m;
  int n;
  int k;
  int gemm_k_iterations;
  const half* ptr_A;
  const half* ptr_B;
  half* ptr_C;
  const half* ptr_D;
  const uint32_t* mask_ptr;
  uint32_t mask_filter;
  bool reverse_mask;
  half alpha;
  half beta;
  half act_alpha;
  half act_beta;
  Activation act_type;
  dim3 grid_dims;
  SparseParams itera_params_;
  WeightOptParams iterb_params_;
  OutIteratorParams out_params_;
  OutIteratorParams out_params_source_;
  /**
   * @param problem 
   * @param A 
   * @param B 
   * @param C 
   * @param D 
   * @param mask_ptr 
   * @param mask_argsort_ptr 
   * @param indice_ptr 
   * @param mask_filter 
   * @param reverse_mask 
   * @param alpha 
   * @param beta 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   * @param split_k_slices 
   * @param d_is_bias 
   */
  __host__ __device__ ConvParams(ConvProblem problem, const half* A, const half* B, half* C, const half* D, const uint32_t* mask_ptr, const int* mask_argsort_ptr, const int* indice_ptr, uint32_t mask_filter, bool reverse_mask, half alpha, half beta, half act_alpha, half act_beta, Activation act_type, int split_k_slices, bool d_is_bias) : problem(problem), itera_params_(problem, indice_ptr, mask_argsort_ptr), iterb_params_(problem, LayoutB::from_shape(problem.get_weight_shape())), ptr_A(A), ptr_B(B), ptr_C(C), ptr_D(D), mask_ptr(mask_ptr), mask_filter(mask_filter), reverse_mask(reverse_mask), alpha(alpha), beta(beta), act_alpha(act_alpha), act_beta(act_beta), act_type(act_type)  {
  
    auto mnk = problem.implicit_gemm_mnk();
    m = mnk[0];
    n = mnk[1];
    k = mnk[2];
    gemm_k_iterations = problem.implicit_gemm_k_iterations(32);
    #if !defined(__CUDACC_RTC__) && !defined(__NVCC__)
    assert(gemm_k_iterations % problem.kernel_volume == 0 && "error");
    #endif
    auto grid_dims_arr = ConvUtils::get_spconv_logical_tile_count(m, n, k, 
        64, 128, split_k_slices, problem.kernel_volume);
    grid_dims.x = grid_dims_arr[0];
    grid_dims.y = grid_dims_arr[1];
    grid_dims.z = grid_dims_arr[2];
    gemm_k_iterations /= problem.kernel_volume;
    itera_params_.set_inc_reset_for_inc_k_first(gemm_k_iterations);
    iterb_params_.set_inc_reset_for_inc_k_first(gemm_k_iterations);
    out_params_ = OutIteratorParams(n, mask_argsort_ptr);
    out_params_source_ = OutIteratorParams(d_is_bias ? 0 : n, d_is_bias ? nullptr : mask_argsort_ptr);
  }
};

} // namespace conv
} // namespace cumm