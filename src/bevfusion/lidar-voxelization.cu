/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>

#include "common/check.hpp"
#include "common/launch.cuh"
#include "lidar-voxelization.hpp"

namespace bevfusion {
namespace lidar {

static __device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

static __device__ inline void insert_to_hash_table(const uint32_t key, uint32_t *value, const uint32_t hash_size,
                                                   uint32_t *hash_table) {//将{voxel_index和有效voxel个数组成hash表保存下来，其中有效voxel个数是逐个+1的原子操作}
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  while (true) {
    uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key);
    if (pre_key == empty_key) {//以前没有填充过这个体素栅格
      hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1);//填充并有效体素个数+1
      break;
    } else if (pre_key == key) {//以前填充过，即有多个点云投影到了同一个voxel的情况下，不重复填充hash
      break;
    }
    slot = (slot + 1) % (hash_size / 2);
  }
}

static __device__ inline uint32_t lookup_hash_table(const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (true /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size / 2];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % (hash_size / 2);
    }
  }
  return empty_key;
}

static __global__ void build_hash_table_kernel(size_t points_size, const half *points, VoxelizationParameter param,
                                               unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = cuda_linear_index;//每个激光点分配一个thread
  if (point_idx >= points_size) return;//共计分配points_size个thread

  float px = points[param.num_feature * point_idx];//取当前激光点的xyz信息
  float py = points[param.num_feature * point_idx + 1];
  float pz = points[param.num_feature * point_idx + 2];

  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);//计算voxel x坐标
  if (voxel_idx < 0 || voxel_idx >= param.grid_size.x) return;//不在范围内舍去

  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  if (voxel_idy < 0 || voxel_idy >= param.grid_size.y) return;

  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if (voxel_idz < 0 || voxel_idz >= param.grid_size.z) return;
  unsigned int voxel_offset = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx;//将三维体素坐标转换为一维index
  insert_to_hash_table(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);//将voxel_index与有效voxel_num组成hash保存在hash_table中
}

template <CoordinateOrder order>
static __device__ void save_result_by_order(uint4 *output, uint x, uint y, uint z);

template <>
__device__ void save_result_by_order<CoordinateOrder::XYZ>(uint4 *output, uint x, uint y, uint z) {
  *output = make_uint4(0, x, y, z);
}

template <>
__device__ void save_result_by_order<CoordinateOrder::ZYX>(uint4 *output, uint x, uint y, uint z) {
  *output = make_uint4(0, z, y, x);
}

template <CoordinateOrder order>
static __global__ void voxelization_kernel(size_t points_size, const half *points, VoxelizationParameter param,
                                           unsigned int *hash_table, unsigned int *num_points_per_voxel, float *voxels_temp,
                                           unsigned int *voxel_indices) {
  int point_idx = cuda_linear_index;//仍然是每个激光点分配一个thread
  if (point_idx >= points_size) return;

  float px = points[param.num_feature * point_idx];//取点的xyz坐标
  float py = points[param.num_feature * point_idx + 1];
  float pz = points[param.num_feature * point_idx + 2];

  if (px < param.min_range.x || px >= param.max_range.x || py < param.min_range.y || py >= param.max_range.y ||
      pz < param.min_range.z || pz >= param.max_range.z) {//点不在设置的有效检测范围内的话不处理
    return;
  }

  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);//计算voxel index
  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if ((voxel_idx < 0 || voxel_idx >= param.grid_size.x)) {//voxel index不在栅格范围内也不处理
    return;
  }
  if ((voxel_idy < 0 || voxel_idy >= param.grid_size.y)) {
    return;
  }
  if ((voxel_idz < 0 || voxel_idz >= param.grid_size.z)) {
    return;
  }

  unsigned int voxel_offset = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx;//一维index,zyx顺序???

  // scatter to voxels
  unsigned int voxel_id = lookup_hash_table(voxel_offset, points_size * 2 * 2, hash_table);//id序号，一般从0-d_real_num_voxels_
  if (voxel_id >= param.max_voxels) {
    return;
  }

  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);//num_points_per_voxel + voxel_id即找到当前voxel对应变量，变量内保存的是当前voxel中点的个数，初始化时候为0，现在执行的是+1操作
  if (current_num < param.max_points_per_voxel) {//当前voxel内的点个数小于10的情况下，计算voxel特征，否则舍去这个点的操作
    unsigned int dst_offset = voxel_id * (param.num_feature * param.max_points_per_voxel) + current_num * param.num_feature;
    unsigned int src_offset = point_idx * param.num_feature;
    for (int feature_idx = 0; feature_idx < param.num_feature; ++feature_idx) {
      voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];//将当前点特征保存到对应voxels_temp中
    }

    // now only deal with batch_size = 1
    // since not sure what the input format will be if batch size > 1
    save_result_by_order<order>(&((uint4 *)voxel_indices)[voxel_id], voxel_idx, voxel_idy, voxel_idz);//将当前voxel的坐标保存下来到voxel_indices中,保存为uint4[0, x, y, z],多个点对应同一个voxel的话，会被反复赋值多次
  }
}

static __global__ void reduce_mean_kernel(size_t num_voxels, float *voxels_temp, unsigned int *num_points_per_voxel,
                                          int max_points_per_voxel, int feature_num, half *voxel_features) {
  int voxel_idx = cuda_linear_index;//每个有效voxel分配一个thread
  if (voxel_idx >= num_voxels) return;

  num_points_per_voxel[voxel_idx] =
      num_points_per_voxel[voxel_idx] > max_points_per_voxel ? max_points_per_voxel : num_points_per_voxel[voxel_idx];//将当前voxel中的点的个数限制在10以内
  int valid_points_num = num_points_per_voxel[voxel_idx];//当前voxel点个数
  int offset = voxel_idx * max_points_per_voxel * feature_num;
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {//逐channel（5）
    for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
      voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];//将所有点的特征叠加到第一个点的位置上去
    }
    voxels_temp[offset + feature_idx] /= valid_points_num;//求平均
  }

  // move to be continuous
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    int dst_offset = voxel_idx * feature_num;
    int src_offset = voxel_idx * feature_num * max_points_per_voxel;
    voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);//保存在变量voxel_features中
  }
}

nvtype::Int3 VoxelizationParameter::compute_grid_size(const nvtype::Float3 &max_range, const nvtype::Float3 &min_range,
                                                      const nvtype::Float3 &voxel_size) {
  nvtype::Int3 size;
  size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
  size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
  size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));
  return size;
}

class VoxelizationImplement : public Voxelization {
 public:
  virtual ~VoxelizationImplement() {
    if (hash_table_) checkRuntime(cudaFree(hash_table_));
    if (voxels_temp_) checkRuntime(cudaFree(voxels_temp_));

    if (d_voxel_features_) checkRuntime(cudaFree(d_voxel_features_));
    if (d_voxel_num_) checkRuntime(cudaFree(d_voxel_num_));
    if (d_voxel_indices_) checkRuntime(cudaFree(d_voxel_indices_));

    if (d_real_num_voxels_) checkRuntime(cudaFree(d_real_num_voxels_));
    if (h_real_num_voxels_) checkRuntime(cudaFreeHost(h_real_num_voxels_));
    if (h_voxel_features_) checkRuntime(cudaFreeHost(h_voxel_features_));
    if (h_voxel_indices_) checkRuntime(cudaFreeHost(h_voxel_indices_));

  }

  bool init(VoxelizationParameter param) {//初始化
    this->param_ = param;
    this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};//体素栅格大小

    this->hash_table_size_ = param_.max_points * 2 * 2 * sizeof(unsigned int);//?
    this->voxels_temp_size_ = param_.max_voxels * param_.max_points_per_voxel * param_.num_feature * sizeof(float);//体素栅格保存最大字节个数：16w * 10 * 5 * 4
    this->voxel_features_size_ = param_.max_voxels * param_.max_points_per_voxel * param_.num_feature * sizeof(half);//栅格特征保存最大字节个数，用half数据个数存储
    this->voxel_num_size_ = param_.max_voxels * sizeof(unsigned int);//
    this->voxel_idxs_size_ = param_.max_voxels * 4 * sizeof(unsigned int);//每个voxel分配了4个uint

    checkRuntime(cudaMalloc(&hash_table_, hash_table_size_));
    checkRuntime(cudaMalloc(&voxels_temp_, voxels_temp_size_));
    checkRuntime(cudaMalloc(&d_voxel_features_, voxel_features_size_));
    checkRuntime(cudaMalloc(&d_voxel_num_, voxel_num_size_));
    checkRuntime(cudaMalloc(&d_voxel_indices_, voxel_idxs_size_));//voxel的坐标：batch x y z
    checkRuntime(cudaMalloc(&d_real_num_voxels_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&h_real_num_voxels_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&h_voxel_features_, voxel_features_size_));//pinned_memory
    checkRuntime(cudaMallocHost(&h_voxel_indices_, voxel_idxs_size_)); 
    
    return true;
  }

  // points and voxels must be of half type
  virtual void forward(const nvtype::half *points, int num_points, void *stream, CoordinateOrder output_order) override {//输入点云、点的个数、坐标顺序(xyz)
    cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
    const half *_points = reinterpret_cast<const half *>(points);
    checkRuntime(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, _stream));
    checkRuntime(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, _stream));
    checkRuntime(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, _stream));
    checkRuntime(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), _stream));
    cuda_linear_launch(build_hash_table_kernel, _stream, num_points, _points, param_, hash_table_, d_real_num_voxels_);//将有效voxel的个数保存在d_real_num_voxels_中,将voxel_index与real_num对应保存在hash_table_中
    checkRuntime(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, _stream));//将active num拷贝到cpu内存变量上：h_real_num_voxels_

    // for difference output order
    if (output_order == CoordinateOrder::XYZ) {//bevfusion用的是xyz
      cuda_linear_launch(voxelization_kernel<CoordinateOrder::XYZ>, _stream, num_points, _points, param_, hash_table_,
                         d_voxel_num_, voxels_temp_, d_voxel_indices_);//逐个将点的特征存放在对应的voxel中，保存在voxels_temp_中，并将当前的voxel坐标保存在d_voxel_indices_中
      this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};
    } else if (output_order == CoordinateOrder::ZYX) {
      cuda_linear_launch(voxelization_kernel<CoordinateOrder::ZYX>, _stream, num_points, _points, param_, hash_table_,
                         d_voxel_num_, voxels_temp_, d_voxel_indices_);
      this->output_grid_size_ = {(int)param_.grid_size.z + 1, (int)param_.grid_size.y, (int)param_.grid_size.x};
    } else
      Assertf(false, "Invalid output_order: %d", static_cast<int>(output_order));

    checkRuntime(cudaStreamSynchronize(_stream));

    real_num_voxels_ = *h_real_num_voxels_;
    cuda_linear_launch(reduce_mean_kernel, _stream, real_num_voxels_, voxels_temp_, d_voxel_num_, param_.max_points_per_voxel,
                       param_.num_feature, d_voxel_features_);//将voxels_temp_中的点特征按照维度求平均，保存在d_voxel_features_中
    //内存拷贝
    // checkRuntime(cudaMemcpyAsync(h_voxel_features_, d_voxel_features_, voxel_features_size_, cudaMemcpyDeviceToHost, _stream));
    // checkRuntime(cudaMemcpyAsync(h_voxel_indices_, d_voxel_indices_, voxel_idxs_size_, cudaMemcpyDeviceToHost, _stream));
    
    // checkRuntime(cudaStreamSynchronize(_stream));
  }

  virtual unsigned int num_voxels() override { return real_num_voxels_; }//非空体素栅格个数

  virtual unsigned int voxel_dim() override { return param_.num_feature; }//体素特征维度:5,xyzi?

  virtual unsigned int indices_dim() override { return 4; }//batch x y z

  virtual std::vector<int> grid_size() override { return output_grid_size_; }//[1440, 1440, 41]

  virtual const void *indices() override { return d_voxel_indices_; }//每个voxel的坐标(indice),uint4*real_num_voxels_,保存在cuda上

  virtual const void *host_indices() override { return h_voxel_indices_; }

  virtual const void *features() override { return d_voxel_features_; }//有效voxel特征，5*real_num_voxels_，保存在cuda上

  virtual const void *host_features() override { return h_voxel_features_; }

  virtual CoordinateOrder order() override { return order_; }

 private:
  CoordinateOrder order_ = CoordinateOrder::NoneOrder;
  VoxelizationParameter param_;
  unsigned int real_num_voxels_ = 0;//有效voxel的个数 = *h_real_num_voxels_
  std::vector<int> output_grid_size_;//grid维度

  unsigned int *hash_table_ = nullptr;//voxel_index-voxel_num 的hash
  float *voxels_temp_ = nullptr;//voxel特征临时存放，保存的是每个voxel中的所有点(<10)
  unsigned int *d_real_num_voxels_ = nullptr;//gpu上的变量，存储有效voxel个数
  unsigned int *h_real_num_voxels_ = nullptr;//cpu上的变量，存储的是有效voxel的个数，为啥用指针啊？
  unsigned int *d_voxel_num_ = nullptr;//每个voxel中的点的个数
  half *d_voxel_features_ = nullptr;//voxel特征(voxel内的所有点求平均之后的)
  half *h_voxel_features_ = nullptr;//存放d_voxel_features_的拷贝，在cpu内存上
  unsigned int *d_voxel_indices_ = nullptr;//保存每个voxel的坐标(indice)，uint4*real_num_voxels_
  unsigned int *h_voxel_indices_ = nullptr;
  unsigned int hash_table_size_;
  unsigned int voxels_temp_size_;
  unsigned int voxel_features_size_;
  unsigned int voxel_idxs_size_;
  unsigned int voxel_num_size_;
};

std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter param) {
  std::shared_ptr<VoxelizationImplement> impl(new VoxelizationImplement());
  if (!impl->init(param)) {//
    impl.reset();
  }
  return impl;
}

};  // namespace lidar
};  // namespace bevfusion
