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

// 网格体积所需位数: 索引范围 [0, vol), 需 2^bits >= vol, 即 bits = ceil(log2(vol))。
// 注意不能用 floor(log2): 少 1 位会漏掉最高位索引, 排序错序 (实测 [720,720,21] 返回 23
// 而应 24, 导致检测异常)。
inline int voxel_index_bits(const std::vector<int>& grid) {
  int64_t vol = 1;
  for (int d : grid) vol *= d;
  int bits = 0;
  int64_t cap = 1;
  while (cap < vol) {
    cap <<= 1;
    ++bits;
  }
  return bits;
}

// bits: 排序键有效位数 (voxel 索引=网格体积位数, mask= kernelVolume),
// 传 0 表示 32 位全量排序。限位可减少 cub radix sort 的 pass 数。
nv::Tensor find_unique_elements_cuda(nv::Tensor& src_tensor, int bits, void* stream);

int generate_subm_conv_inds(nv::Tensor indices, nv::Tensor hashdata_k,
                            nv::Tensor hashdata_v, nv::Tensor indice_pairs,
                            std::vector<int> input_dims, std::vector<int> ksize,
                            nv::Tensor indice_pair_mask,
                            ConvOutLocIter& loc_iter, void* stream);

nv::Tensor sort_1d_by_key_allocator_v2(nv::Tensor data, nv::Tensor indices, int bits, void* stream);

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

// ===== direct_table (线性哈希表) 相关, 取代 sort+unique+二分 构建 stride 规则簿 =====
// stage1: 清哈希表 + 清 (kv,input) 数组 + 哈希插入去重 (kernel 见 indice.cu.h)
void generate_conv_inds_mask_stage1_direct_table(nv::Tensor indices,
                                                 nv::Tensor hash_k, nv::Tensor hash_v,
                                                 nv::Tensor indice_pairs_uniq_bkp,
                                                 std::vector<int> ksize,
                                                 ConvOutLocIter& loc_iter,
                                                 void* stream);
// unique_hash: 收集表项 -> indice_pairs_uniq, 返回 num_act_out (host 读回, 内部同步)
int unique_hash_table(nv::Tensor hash_k, nv::Tensor hash_v,
                      nv::Tensor indice_pairs_uniq, void* stream);
// 解码 unique 输出 1D 索引 -> out_inds 坐标
void assign_output_direct_hash(nv::Tensor out_inds, nv::Tensor indice_pairs_uniq,
                               int num_act_out, ConvOutLocIter& loc_iter,
                               void* stream);
// stage2: O(1) 查表填 mask_fwd + indice_pairs_fwd
void generate_conv_inds_stage2_mask_direct_table(nv::Tensor indices,
                                                 nv::Tensor hash_k, nv::Tensor hash_v,
                                                 nv::Tensor indice_pairs,
                                                 nv::Tensor indice_pairs_uniq_bkp,
                                                 nv::Tensor mask_fwd,
                                                 std::vector<int> ksize,
                                                 void* stream);
} // namespace spconv

#endif