#ifndef __SPCONV_NORMAL_OP_ADD_HPP__
#define __SPCONV_NORMAL_OP_ADD_HPP__
#include "tensor.hpp"

namespace spconv {

void add_cuda(nv::Tensor features0, nv::Tensor features1, nv::Tensor output,
              int64_t act_num, int64_t voxel_dim, void* stream, bool relu = false);


}// namespace spconv

#endif