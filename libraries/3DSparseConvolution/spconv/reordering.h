// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPARSE_REORDERING_FUNCTOR_H_
#define SPARSE_REORDERING_FUNCTOR_H_
#include "tensor.hpp"

namespace spconv {
// bias 融合版: conv epilogue 直接完成 D = alpha*Accum + bias[k] + ReLU
// (ConvParams 的 d_is_bias 路径: ConstOutIterator stride=0 按 K 维广播读 bias),
// 省去单独的 addBiasAndRelu kernel 与一次中间 tensor 读写。
// bias 为空 tensor 时不加 bias (beta=0); relu 控制是否接 ReLU。
void implicit_gemm_cuda(nv::Tensor features,
                        nv::Tensor filters,
                        nv::Tensor pair_fwd,
                        nv::Tensor pair_mask_fwd,
                        nv::Tensor mask_argsort_fwd,
                        nv::Tensor out_features,
                        nv::Tensor bias,
                        bool relu,
                        void* stream);
} // namespace spconv

#endif