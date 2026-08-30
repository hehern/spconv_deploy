#pragma once
#include "ConvProblem.h"
namespace cumm {
namespace conv {

struct SparseParams {
  int filter_c_delta;
  int inc_c_next;
  int inc_c_reset;
  int const* indice_ptr_;
  int const* mask_argsort_ptr_;
  int RS;
  __forceinline__ __host__ __device__  SparseParams(ConvProblem const& problem, int const* indice_ptr, int const* mask_argsort_ptr) : indice_ptr_(indice_ptr), mask_argsort_ptr_(mask_argsort_ptr)  {
    
    RS = problem.kernel_volume;
    filter_c_delta = 32 * problem.split_k_slices;
    inc_c_next = filter_c_delta * 2 ;
  }
  __forceinline__ __host__ __device__ void set_inc_reset_for_inc_k_first(int gemm_iters_k)   {
    
    inc_c_reset = (- filter_c_delta) * gemm_iters_k * 2  ;
  }
};

} // namespace conv
} // namespace cumm