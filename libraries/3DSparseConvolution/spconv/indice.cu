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

__global__ void printNumber(int *number) {
    // 假设我们只打印一个线程
    printf("Number on GPU: %d, %d, %d\n", number[0], number[1], number[2]);
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  gridsOut:nv::Tensor, shape:{outputVolume},需要计算的变量
  indicePairs:nv::Tensor, shape:{2,27,n},需要计算的量
  indiceNum:nv::Tensor, shape:{27},需要计算的量
*/
int create_submconv_indice_pair_cuda(
    nv::Tensor indicesIn, nv::Tensor gridsOut, nv::Tensor indicePairs,
    nv::Tensor indiceNum, nv::Tensor outSpatialShape, int spatialVolume,
    void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  auto ndim = outSpatialShape.size(0);//3,三维的体素栅格
  auto numActIn = indicesIn.size(0);//active voxel num: n

  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return 0;

  int* indicesIn_ptr = indicesIn.ptr<int>();//(batch,x,y,z)
  int* gridsOut_ptr = gridsOut.ptr<int>();
  int* outSpatialShape_ptr = outSpatialShape.ptr<int>();//size:xyz
  int* indiceNum_ptr = indiceNum.ptr<int>();//size:kernelVolume=27
  int* indicePairs_ptr = indicePairs.ptr<int>();//size:{2,27,n}

  cuda_linear_launch(prepareSubMGridKernel, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, outSpatialShape_ptr, spatialVolume);//计算Hash_out：建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  cuda_linear_launch(getSubMIndicePairsKernel3, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, indicePairs_ptr, indiceNum_ptr, outSpatialShape_ptr, spatialVolume, kernelVolume);//计算rule_book,保存在indicePairs_ptr和indiceNum_ptr中
  return numActIn;
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  indicePairs:shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是grid的一维index,需要计算的量
  indiceNum:nv::Tensor, shape:{27},需要计算填充的量
  indicePairUnique:nv::Tensor, shape:{N*27+1},需要计算的量
  kernelSize: 3,3,3
  stride: eg:2,2,2
  padding: eg:1, 1, 1
  dilation: eg:1, 1, 1
  outSpatialShape: eg:{720, 720, 21}
  spatialVolume: 为outSpatialShape的累乘
*/
int create_conv_indice_pair_p1_cuda(
    nv::Tensor indicesIn, nv::Tensor indicePairs, nv::Tensor indiceNum,
    nv::Tensor indicePairUnique, std::vector<int> kernelSize,
    std::vector<int> stride, std::vector<int> padding,
    std::vector<int> dilation, std::vector<int> outSpatialShape,
    int spatialVolume, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = kernelSize.size();//3
  auto numActIn = indicesIn.size(0);//active voxel num: n
  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)
    return 0;

  nv::Tensor ks = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);//
  nv::Tensor st = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor pa = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor di = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ks.ptr<int>(), kernelSize.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(st.ptr<int>(), stride.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(pa.ptr<int>(), padding.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(di.ptr<int>(), dilation.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));


  cuda_linear_launch(prepareIndicePairsKernel, _stream, numActIn, indicesIn.ptr<int>(), indicePairs.ptr<int>(), 
      indiceNum.ptr<int>(), indicePairUnique.ptr<int>(), 
      ks.ptr<int>(), st.ptr<int>(), pa.ptr<int>(), di.ptr<int>(),
      ou.ptr<int>(), spatialVolume, kernelVolume);
  return 1;
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)--输入数据的
  indicesOut:{numAct * kernelVolume, coorDim + 1}需要计算的输出数据的坐标!!!
  gridsOut:outputVolume形状的grid，对应的输出有效位置存放的是有效voxel序号，从0-numAct
  indicePairs:shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout,这里需要先根据里面之前存的输出一维index再根据grid_out填充vout
  indiceNum:nv::Tensor, shape:{27},conv每个元素参与计算的次数，即count
  indicePairUnique:nv::Tensor, shape:{numActOut},里面保存的是输出grid中有效voxel的坐标（一维的）
  outSpatialShape: eg:{720, 720, 21}
*/
int create_conv_indice_pair_p2_cuda(
    nv::Tensor indicesIn, nv::Tensor indicesOut, nv::Tensor gridsOut,
    nv::Tensor indicePairs, nv::Tensor indiceNum, nv::Tensor indicePairUnique, 
    std::vector<int> outSpatialShape, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = outSpatialShape.size();//3
  int numActIn = indicesIn.size(0);//active voxel num: n
  int numAct = indicePairUnique.size(0) - 1;//不重复输出序号个数-1,去掉初始化时候的max值

  auto kernelVolume = indiceNum.size(0);
  if (numActIn == 0)
    return 0;

  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  
  cuda_linear_launch(assignGridAndIndiceOutKernel, _stream, numAct, indicesOut.ptr<int>(), gridsOut.ptr<int>(), 
    indicePairUnique.ptr<int>(), ou.ptr<int>());//计算indicesOut
  cuda_linear_launch(assignIndicePairsKernel, _stream, numActIn, gridsOut.ptr<int>(),
    indicePairs.ptr<int>(), kernelVolume);//填充indicePairs中的vout

  return numAct;
}

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

void judgeIndicesOutshape(nv::Tensor indices,
                          std::vector<int> outSpatialShape,
                          void* stream) {
  // 判断indices值是否在outSpatialShape范围内
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = outSpatialShape.size();//3
  auto numActIn = indices.size(0);//active voxel num: n

  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return;

  int* indices_ptr = indices.ptr<int>();//(batch,x,y,z)
  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));

  cuda_linear_launch(judgeIndicesOutshapeKernel, _stream, numActIn, indices_ptr, ou.ptr<int>());
 return;
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