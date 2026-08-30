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

#ifndef SPARSE_CONV_INDICE_FUNCTOR_H_
#define SPARSE_CONV_INDICE_FUNCTOR_H_
#include "tensor.hpp"
#include "conv/ConvOutLocIter.h"

namespace spconv {

#define divup(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

int create_conv_indice_pair_p1_cuda(
    nv::Tensor indicesIn, nv::Tensor indicePairs, nv::Tensor indiceNum,
    nv::Tensor indicePairUnique, std::vector<int> kernelSize,
    std::vector<int> stride, std::vector<int> padding,
    std::vector<int> dilation, std::vector<int> outSpatialShape,
    int spatialVolume, void* stream);

int create_conv_indice_pair_p2_cuda(
    nv::Tensor indicesIn, nv::Tensor indicesOut, nv::Tensor gridsOut,
    nv::Tensor indicePairs, nv::Tensor indiceNum,
    nv::Tensor indicePairUnique, std::vector<int> outSpatialShape,
    void* stream);

int create_submconv_indice_pair_cuda(
    nv::Tensor indicesIn, nv::Tensor gridsOut, nv::Tensor indicePairs,
    nv::Tensor indiceNum, nv::Tensor outSpatialShape, int spatialVolume,
    void* stream);

nv::Tensor find_unique_elements_cuda(nv::Tensor& src_tensor, void* stream);

void judgeIndicesOutshape(nv::Tensor indices, std::vector<int> outSpatialShape, void* stream);

int generate_subm_conv_inds(nv::Tensor indices, nv::Tensor hashdata_k, 
                            nv::Tensor hashdata_v, nv::Tensor indice_pairs,
                            std::vector<int> input_dims, std::vector<int> ksize, 
                            nv::Tensor indice_pair_mask, 
                            ConvOutLocIter& loc_iter, void* stream);

nv::Tensor sort_1d_by_key_allocator_v2(nv::Tensor data, nv::Tensor indices, void* stream);

void generate_conv_inds_mask_stage1(nv::Tensor indices, 
                                    nv::Tensor indice_pairs_uniq,
                                    std::vector<int> ksize,
                                    ConvOutLocIter& loc_iter,
                                    void* stream);

int generate_conv_inds_mask_stage2(nv::Tensor indices, 
                                   nv::Tensor hashdata_k, 
                                   nv::Tensor hashdata_v, 
                                   nv::Tensor indice_pairs,
                                   nv::Tensor indice_pairs_uniq, 
                                   nv::Tensor indice_pairs_uniq_before_sort, 
                                   nv::Tensor out_inds, 
                                   nv::Tensor mask_fwd,
                                   int num_out_act,
                                   std::vector<int> ksize, 
                                   ConvOutLocIter& loc_iter,
                                   void* stream);
} // namespace spconv

#endif