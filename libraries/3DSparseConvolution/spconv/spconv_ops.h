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

#ifndef SPARSE_CONV_OP_H_
#define SPARSE_CONV_OP_H_

#include "tensor.hpp"
#include "indice.h"
#include "reordering.h"

namespace spconv {


std::vector<nv::Tensor>
getIndicePairs(nv::Tensor indices,
               std::vector<int> outSpatialShape,
               std::vector<int> spatialShape,
               std::vector<int> kernelSize, std::vector<int> stride,
               std::vector<int> padding, std::vector<int> dilation,
               bool subM, void* stream);

nv::Tensor indiceConv(nv::Tensor features, 
                      nv::Tensor filters,
                      nv::Tensor indicePairs, 
                      nv::Tensor indiceNum,
                      int64_t numActOut,
                      bool subM, void* stream);

nv::Tensor indiceConv2(nv::Tensor features, 
                       nv::Tensor filters,
                       nv::Tensor indicePairs, 
                       nv::Tensor indiceNum,
                       int64_t numActOut,
                       bool subM,
                       const std::string &rulebook,
                       void* stream);

void printFeatures(nv::Tensor features, void* stream);
void clear_indice_cache();

std::vector<nv::Tensor>
getIndicePairsImplicitGemm(nv::Tensor indices,
                           std::vector<int> outSpatialShape,
                           std::vector<int> spatialShape,
                           std::vector<int> kernelSize,
                           std::vector<int> stride,
                           std::vector<int> padding,
                           std::vector<int> dilation,
                           bool subm,
                           void* stream);

} // namespace spconv
#endif