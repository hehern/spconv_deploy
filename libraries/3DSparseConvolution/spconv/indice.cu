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

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <iostream>
#include "spconv/indice.cu.h"
#include "spconv/indice.h"

namespace spconv {


nv::Tensor find_unique_elements_cuda(nv::Tensor& src_tensor, void* stream) {

  int64_t num = src_tensor.shape[0];
  if (num == 0) {
      return nv::Tensor::create(std::vector<int64_t>{0}, nv::DataType::Int32);
  }

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int* begin = src_tensor.ptr<int>();
  int* end = begin + num;

  // stream-aware sort + unique: 用thrust::cuda::par.on避免默认流同步
  auto policy = thrust::cuda::par.on(_stream);
  thrust::sort(policy, begin, end);
  int* unique_end = thrust::unique(policy, begin, end);

  checkRuntime(cudaStreamSynchronize(_stream));

  int64_t unique_count = unique_end - begin;
  return nv::Tensor::from_data(
    begin, std::vector<int64_t>{unique_count}, nv::DataType::Int32);
}


/*
  填充indice_pairs以及indice_pair_mask
  in:
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  hashdata_k:nv::Tensor, shape:{numAnum_voxels}
  hashdata_v:nv::Tensor, shape:{numAnum_voxels}
  indice_pairs:nv::Tensor, shape:{kernelVolume, numActIn}
  indice_num_per_loc:nv::Tensor, shape:{kernelVolume},每个卷积核元素对应的有效voxel数量
  input_dims: eg:{720, 720, 21}
  ksize: eg:{3, 3, 3}
  dilation: eg:{1, 1, 1}
  indice_pair_mask:nv::Tensor, shape:{numActIn}
*/
int generate_subm_conv_inds(nv::Tensor indices, nv::Tensor hashdata_k,
                            nv::Tensor hashdata_v, nv::Tensor indice_pairs,
                            std::vector<int> input_dims, std::vector<int> ksize,
                            nv::Tensor indice_pair_mask,
                            ConvOutLocIter& loc_iter, void* stream) {

  int numActIn = indices.shape[0];
  if (numActIn == 0) {
    return 0;
  }
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());

  int* indicesIn_ptr = indices.ptr<int>();//(batch,x,y,z)
  int* hashdata_k_ptr = hashdata_k.ptr<int>();
  int* hashdata_v_ptr = hashdata_v.ptr<int>();
  int* indice_pairs_ptr = indice_pairs.ptr<int>();
  int64_t NDim = ksize.size();//3
  nv::Tensor ou = nv::Tensor::create(std::vector<int64_t>{NDim}, nv::DataType::Int32);//output_shape
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), input_dims.data(), input_dims.size()*sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  int* inSpatialShape_ptr = ou.ptr<int>();//size:xyz
  
  cuda_linear_launch(buildSubmConvHashTable, _stream, numActIn, indicesIn_ptr, hashdata_k_ptr, hashdata_v_ptr, inSpatialShape_ptr);//计算Hash_out：建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  // hash 表按 key 升序排序 (value 跟随), 使 calc_subm_conv_indices_mask 可用二分查找。
  // 原实现为无序表 + 线性扫描, 查找复杂度 O(N^2 * RS) (subm2 层约 780 亿次比较/帧),
  // 排序 + 二分后降为 O(N log N + N * RS * log N), 是 Lidar Backbone 的主要耗时来源。
  {
    thrust::device_ptr<int> ptr_sort_k(hashdata_k_ptr);
    thrust::device_ptr<int> ptr_sort_v(hashdata_v_ptr);
    auto thrust_ctx = thrust::cuda::par.on(_stream);
    thrust::sort_by_key(thrust_ctx, ptr_sort_k, ptr_sort_k + numActIn, ptr_sort_v);
  }
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "buildSubmConvHashTable!" << std::endl;
  uint32_t* indice_pair_mask_ptr = indice_pair_mask.ptr<uint32_t>();
  cuda_linear_launch(fill_kernel<uint32_t>, _stream, numActIn, indice_pair_mask_ptr, 1 << (kv / 2));//每个active voxel都与kernel中心位置参与卷积，所以初始化1<<13
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "fill_kernel!" << std::endl;
  dim3 __threads__(std::min(numActIn, 1024));
  dim3 __blocks__(divup(numActIn, std::min(numActIn, 1024)), (kv / 2) + 1);
  calc_subm_conv_indices_mask<<<__blocks__, __threads__, 0, _stream>>>(hashdata_k_ptr, hashdata_v_ptr, indicesIn_ptr,
        indice_pairs_ptr, indice_pair_mask_ptr, numActIn, kv, (kv / 2) + 1, loc_iter);
  // checkRuntime(cudaStreamSynchronize(_stream));
  // std::cout << "calc_subm_conv_indices_mask!" << std::endl;
  return indices.shape[0];
}

nv::Tensor sort_1d_by_key_allocator_v2(nv::Tensor data,
                                       nv::Tensor indices,
                                       void* stream) {

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int numActIn = indices.numel;
  if (numActIn == 0) {
    return indices;
  }
  int32_t* indicesIn_ptr = indices.ptr<int32_t>();
  uint32_t* data_ptr = data.ptr<uint32_t>();
  cuda_linear_launch(arange_kernel<int32_t>, _stream, numActIn, indicesIn_ptr);//0-numActIn-1

  thrust::device_ptr<uint32_t> ptr_tr(data_ptr);
  thrust::device_ptr<int32_t> ptr_k(indicesIn_ptr);
  auto thrust_ctx = thrust::cuda::par.on(_stream);
  thrust::sort_by_key(thrust_ctx, ptr_tr, ptr_tr + numActIn, ptr_k);//按照mask的大小顺序升序排列indicesIn_ptr

  return indices;
}

void generate_conv_inds_mask_stage1(nv::Tensor indices, 
                                    nv::Tensor indice_pairs_uniq,
                                    std::vector<int> ksize,
                                    ConvOutLocIter& loc_iter,
                                    void* stream) {
  
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  int num_act_in = indices.shape[0];
  if (num_act_in == 0) {
    return;
  }
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int64_t uniq_size = indice_pairs_uniq.shape[0];
  int32_t* indice_pairs_uniq_ptr = indice_pairs_uniq.ptr<int32_t>();
  cuda_linear_launch(clean_indices_uniq<int32_t>, _stream, uniq_size, indice_pairs_uniq_ptr);
  // indice_pairs_uniq.fill<int32_t>(std::numeric_limits<int32_t>::max());//这两种写法都可以

  int* indicesIn_ptr = indices.ptr<int>();//(batch,x,y,z)

  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);
  calc_conv_indices_stage1_mask<<<__blocks__, __threads__, 0, _stream>>>(indicesIn_ptr,
    indice_pairs_uniq_ptr, num_act_in, kv, loc_iter);
}

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
                                   void* stream) {

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  int num_act_in = indices.shape[0];
  if (num_act_in == 0 || num_out_act == 0) {
    return num_out_act;
  }
  int* hashdata_k_ptr = hashdata_k.ptr<int>();
  int* hashdata_v_ptr = hashdata_v.ptr<int>();
  int* indice_pairs_ptr = indice_pairs.ptr<int>();
  int* indice_pairs_uniq_ptr = indice_pairs_uniq.ptr<int>();
  int* indice_pairs_uniq_before_sort_ptr = indice_pairs_uniq_before_sort.ptr<int>();
  int* out_inds_ptr = out_inds.ptr<int>();
  uint32_t* mask_fwd_ptr = mask_fwd.ptr<uint32_t>();

  cuda_linear_launch(build_conv_hash_table, _stream, num_out_act, hashdata_k_ptr, hashdata_v_ptr, 
    out_inds_ptr, indice_pairs_uniq_ptr, loc_iter);
  
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  dim3 __threads__(std::min(num_act_in, 1024));
  dim3 __blocks__(divup(num_act_in, std::min(num_act_in, 1024)), kv);

  calc_conv_indices_stage2_inference_mask<<<__blocks__, __threads__, 0, _stream>>>(hashdata_k_ptr, hashdata_v_ptr, indice_pairs_ptr, 
    indice_pairs_uniq_before_sort_ptr, mask_fwd_ptr, num_act_in, num_out_act, kv, hashdata_k.shape[0]);

  return num_out_act;
}

} // namespace spconv