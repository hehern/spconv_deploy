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
void matrix_multiply_cuda(const nv::Tensor& features, const nv::Tensor& filters, nv::Tensor& output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream);
void matrix_multiply_cuda(half* features, const nv::Tensor& filters, half* output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream);
void sparse_gather_cuda(nv::Tensor& buffer, const nv::Tensor& features,
                        const nv::Tensor& indices, int size, int indice_offset,
                        void* stream);
void sparse_scatter_add_cuda(const nv::Tensor& buffer, nv::Tensor& outFeatures,
                             const nv::Tensor& indices, int size, int indice_offset,
                             void* stream);

// 批量处理版本 - 将27次循环合并
void sparse_gather_all_cuda(nv::Tensor& buffer, const nv::Tensor& features,
                            const nv::Tensor& indices, const int* kernelIds,
                            const int* kernelOffsets, int numActIn, int totalCount,
                            void* stream, int kernelVolume);
void sparse_scatter_add_all_cuda(nv::Tensor& buffer, nv::Tensor& output,
                                 const nv::Tensor& indices, const int* kernelIds,
                                 const int* kernelOffsets, int numActIn, int totalCount,
                                 void* stream, int kernelVolume);

void addBiasAndRelu(nv::Tensor features, nv::Tensor bias,
                    bool Relu, void* stream);

void implicit_gemm_cuda(nv::Tensor features,
                        nv::Tensor filters,
                        nv::Tensor pair_fwd,
                        nv::Tensor pair_mask_fwd,
                        nv::Tensor mask_argsort_fwd,
                        nv::Tensor out_features,
                        void* stream);
} // namespace spconv

#endif