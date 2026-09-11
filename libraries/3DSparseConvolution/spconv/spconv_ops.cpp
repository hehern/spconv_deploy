#include <limits>
#include <iostream>
#include <algorithm>
#include <map>
#include <utility>
#include <string>
#include <numeric>


#include <cuda_fp16.h>

#include "spconv_ops.h"
#include "conv/ConvOutLocIter.h"
#include "common/check.hpp"
#include "common/timer.hpp"
namespace spconv {





std::vector<nv::Tensor>
getIndicePairsImplicitGemm(nv::Tensor indices,
                           std::vector<int> outSpatialShape,
                           std::vector<int> spatialShape,
                           std::vector<int> kernelSize,
                           std::vector<int> stride,
                           std::vector<int> padding,
                           std::vector<int> dilation,
                           bool subm,
                           void* stream)   { 

  int kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int>());//kernel volume,eg:27
  ConvOutLocIter loc_iter(kernelSize.data(), stride.data(), padding.data(), dilation.data(), outSpatialShape.data(), spatialShape.data());
  int64_t numAct = indices.shape[0];

  int num_act_out = 0;
  if (subm) {
    num_act_out = numAct;

    // hash_k/v 作为线性哈希表 (generate_subm_conv_inds 的单 cooperative kernel 使用):
    // key=voxel 1D 索引, value=voxel id。按 >=2x numAct 分配, 负载<=0.5 保证线性探测效率
    int hash_size = (int)numAct * 2 + 1;
    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{hash_size}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{hash_size}, nv::DataType::Int32);
    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, numAct}, nv::DataType::Int32);
    indicePairs.fill<int32_t>(-1, stream);
    
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::UInt32);//每个active voxel都与kernel中的哪个元素进行卷积的mask
    generate_subm_conv_inds(indices, hash_k, hash_v, indicePairs,
        spatialShape, kernelSize, pair_mask, loc_iter, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, kernelVolume, stream);//对pair_mask进行排序，返回排序后的索引

    nv::Tensor numActOut = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, false);
    numActOut.ptr<int32_t>()[0] = num_act_out;
    
    return {indices, indicePairs, pair_mask, mask_argsort, numActOut};

  } else {
    // direct_table 分支 (参考 spconv 2.x): 线性哈希表取代 sort+unique+二分。
    // stage1 把输出 voxel 1D 索引哈希插入(去重, 无需排序), unique_hash 收集表项并
    // 赋输出id, stage2 O(1) 查表替代 find_in_hash_k 二分查找。见 indice.cu.h 说明。
    auto pair_size = kernelVolume * numAct;
    // 哈希表大小: 不同输出 <= pair_size (每个 (kv,input) 至多贡献 1 个), 保证有空槽
    int hash_size = (int)pair_size + 1;
    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{hash_size}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{hash_size}, nv::DataType::Int32);
    nv::Tensor indice_pairs_uniq_bkp = nv::Tensor::create(std::vector<int64_t>{pair_size + 1}, nv::DataType::Int32);  // (kv,input)->输出索引
    nv::Tensor indice_pairs_uniq = nv::Tensor::create(std::vector<int64_t>{pair_size + 1}, nv::DataType::Int32);      // 收集的 unique 输出索引

    // stage1: 哈希插入去重 + 填 (kv,input) 数组
    generate_conv_inds_mask_stage1_direct_table(indices, hash_k, hash_v,
        indice_pairs_uniq_bkp, kernelSize, loc_iter, stream);

    // unique_hash: 收集表项 -> indice_pairs_uniq, 返回 num_act_out (内部 DtoH+同步)
    num_act_out = unique_hash_table(hash_k, hash_v, indice_pairs_uniq, stream);

    // 解码 unique 输出索引 -> out_inds 坐标
    nv::Tensor out_inds = nv::Tensor::create(std::vector<int64_t>{num_act_out, indices.shape[1]}, indices.dtype());
    assign_output_direct_hash(out_inds, indice_pairs_uniq, num_act_out, loc_iter, stream);

    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, num_act_out}, indices.dtype());
    indicePairs.fill<int32_t>(-1, stream);
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::UInt32);
    pair_mask.fill<uint32_t>(0, stream);

    // stage2: O(1) 查表填 mask_fwd + indice_pairs_fwd
    generate_conv_inds_stage2_mask_direct_table(indices, hash_k, hash_v,
        indicePairs, indice_pairs_uniq_bkp, pair_mask, kernelSize, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, kernelVolume, stream);
    nv::Tensor numActOut = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, false);
    numActOut.ptr<int32_t>()[0] = num_act_out;
    return {out_inds, indicePairs, pair_mask, mask_argsort, numActOut};
  
  }
}

/*
  indicePairs: shape:{27,n},就是rule_book，存储参与当前kernel位置卷积的active voxel的序号[0, numActOut-1]
  pair_mask: shape:{n},每个active voxel都与kernel中的哪个元素进行卷积的mask
  mask_argsort: shape:{n},对pair_mask进行排序，返回排序后的索引
*/
nv::Tensor
implicit_gemm(nv::Tensor features,
              nv::Tensor filters, //格式为eg:权重[16,3*3*3,5]，输出channel kernel_volume 输入channel
              nv::Tensor indicePairs,
              nv::Tensor pair_mask,
              nv::Tensor mask_argsort,
              int num_activate_out,
              bool is_subm,
              nv::Tensor bias,   // bias 融合进 conv epilogue (空 tensor 表示无 bias)
              bool relu,         // epilogue 是否接 ReLU
              void* stream) {

  int out_channel = filters.shape[0];
  int in_channel = filters.shape[-1];
  int kv = filters.shape[1];

  nv::Tensor out_features;
  if (is_subm) {
    out_features = nv::Tensor::create(std::vector<int64_t>{num_activate_out, out_channel}, features.dtype(), features.device());
  } else {
    out_features = nv::Tensor::create(std::vector<int64_t>{num_activate_out, out_channel}, features.dtype(), features.device());
    out_features.memset(0, stream);
  }

  implicit_gemm_cuda(features, filters, indicePairs, pair_mask, mask_argsort,
                     out_features, bias, relu, stream);

 return out_features;
}

} // namespace spconv
