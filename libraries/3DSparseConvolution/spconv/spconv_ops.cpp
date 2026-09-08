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

    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, numAct}, nv::DataType::Int32);
    indicePairs.fill<int32_t>(-1, stream);
    
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::UInt32);//每个active voxel都与kernel中的哪个元素进行卷积的mask
    generate_subm_conv_inds(indices, hash_k, hash_v, indicePairs,
        spatialShape, kernelSize, pair_mask, loc_iter, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, stream);//对pair_mask进行排序，返回排序后的索引

    nv::Tensor numActOut = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, false);
    numActOut.ptr<int32_t>()[0] = num_act_out;
    
    return {indices, indicePairs, pair_mask, mask_argsort, numActOut};

  } else {
    auto pair_size = kernelVolume * numAct;
    nv::Tensor indice_pairs_uniq = nv::Tensor::create(std::vector<int64_t>{pair_size + 1}, nv::DataType::Int32);

    generate_conv_inds_mask_stage1(indices, indice_pairs_uniq, kernelSize, loc_iter, stream);
    // stage2 依赖 stage1 的 (kv, 输入点) 原始布局, 而 find_unique 是原地 sort+unique,
    // 会破坏该布局 —— 必须先备份, 否则 stage2 读到的坐标全部错位 (rulebook 连接错乱)
    nv::Tensor indice_pairs_uniq_backup = indice_pairs_uniq.clone(stream);
    nv::Tensor indicePairUnique_new = find_unique_elements_cuda(indice_pairs_uniq, stream);//挑出tensor中的独立不重复元素,并按照升序排列，indicePairUnique中保存的是vout即输出voxel grid的一维index
    num_act_out = indicePairUnique_new.shape[0];
    // clean_indices_uniq 用 INT_MAX 初始化整个数组(含 pair_size+1 多出的 1 个元素),
    // sort+unique 后哨兵恒留在尾部且被计入 count —— 排除它, 否则多出 1 个
    // 假输出 voxel (哨兵被 inverse 解码成垃圾坐标, 界内则污染输出, 越界则写坏内存)
    if (num_act_out > 0) num_act_out -= 1;

    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    hash_k.fill<int32_t>(std::numeric_limits<int32_t>::max(), stream);
    nv::Tensor out_inds = nv::Tensor::create(std::vector<int64_t>{num_act_out, indices.shape[1]}, indices.dtype());
    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, num_act_out}, indices.dtype());
    indicePairs.fill<int32_t>(-1, stream);
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::UInt32);
    pair_mask.fill<uint32_t>(0, stream);

    generate_conv_inds_mask_stage2(indices, hash_k, hash_v, indicePairs,
        indicePairUnique_new, indice_pairs_uniq_backup,
        out_inds, pair_mask, num_act_out, kernelSize, loc_iter, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, stream);
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
