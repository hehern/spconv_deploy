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
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include "lidar-scn.hpp"
#include "onnx-parser.hpp"

namespace bevfusion {
namespace lidar {

class SCNImplement : public SCN {
 public:
  bool init(const SCNParameter& param) {
    this->param_ = param;//传递参数，
    voxelization_ = create_voxelization(param_.voxelization);//创建一个体素化对象
    if (voxelization_ == nullptr) return false;

    native_scn_ = spconv::load_engine_from_onnx(param_.model, static_cast<spconv::Precision>(param_.precision));//加载onnx：lidar.backbone.xyz.onnx,返回Engine类型
    return native_scn_ != nullptr;
  }

  virtual const nvtype::half* forward(const nvtype::half* points, unsigned int num_points, void* stream) override {//在gpu上保存的点云、点个数
    voxelization_->forward(points, num_points, stream, param_.order);//点云体素化,输出：有效voxel个数（real_num_voxels_）、每个voxel中点平均特征（d_voxel_features_）、特征voxel对应的每个voxel的xyz index
    // 打印feature和indices
    // auto f_h_ptr = (half*)(voxelization_->host_features());
    // auto size = voxelization_->num_voxels() * voxelization_->voxel_dim();
    // printf("features size = %d\n", size);
    // for(size_t i=0; i<size; i++) {
    //   float f_f = __half2float(f_h_ptr[i]);
    //   printf("%f,", f_f);
    //   if ((i+1)%voxelization_->voxel_dim() == 0) {
    //     printf("\n");
    //   }
    // }
    // printf("--------\n");

    // auto ind_h_ptr = (unsigned int*)(voxelization_->host_indices());
    // auto size_ind = voxelization_->num_voxels() * voxelization_->indices_dim();
    // printf("indices size = %d\n", size);
    // for(size_t i=0; i<size_ind; i++) {
    //   printf("%d,", ind_h_ptr[i]);
    //   if ((i+1)%voxelization_->indices_dim() == 0) {
    //     printf("\n");
    //   }
    // }
    // printf("--------\n");
    // printf("num_voxels = %d\n", voxelization_->num_voxels());
    // 第 0 帧是调用方(main.cpp)的 warmup 帧，不启动 profiler，
    // 保证 nsys profile 只统计非 warmup 帧 native_scn_->forward 的耗时
    // const bool profile_this_frame = frame_++ > 0;
    // if (profile_this_frame) cudaProfilerStart();
    native_scn_->forward(
      std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->voxel_dim()}, nv::DataType::Float16,
      (void*)voxelization_->features(), std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->indices_dim()},
      nv::DataType::Int32, (void*)voxelization_->indices(), voxelization_->grid_size(), stream);
    // if (profile_this_frame) cudaProfilerStop();

    // std::cout << "onnx output size = " << native_scn_->num_output() << std::endl;
    // std::vector<int64_t> output0_shape = native_scn_->output(0)->features().shape;
    // std::cout << "SCNImplement->forward output0_shape = ";
    // for (int i=0; i<output0_shape.size(); i++) {
    //   std::cout << output0_shape[i] << ",";
    // }
    // std::cout << std::endl;

    // auto featuresHost = native_scn_->output(0)->features().to_host(stream);
    // auto f_h_ptr = featuresHost.ptr<half>();
    // printf("features numel = %d\n", featuresHost.numel);
    // for(size_t i=0; i<featuresHost.numel; i++) {
    //   float f_f = __half2float(f_h_ptr[i]);
    //   printf("%f,", f_f);
    //   if ((i+1)%output0_shape[3] == 0) {
    //     printf("\n");
    //   }
    // }
    // printf("\n");
    // 分配 CPU 内存
    // size_t print_num = 100*output0_shape[2]*output0_shape[3];
    // std::vector<half> cpu_data(print_num);
    // const void* gpu_data = native_scn_->output(0)->features().ptr<half>();
    // // 将数据从 GPU 复制到 CPU
    // cudaMemcpy(cpu_data.data(), gpu_data, print_num * sizeof(half), cudaMemcpyDeviceToHost);
    // // 输出数据
    // printf("features numel = %d\n", print_num);
    // size_t count = 0;
    // for(size_t i=0; i<print_num; i++) {
    //   float f_f = __half2float(cpu_data[i]);
    //   if (f_f != 0.0f) {
    //     printf("%zu %f,\n", i, f_f);
    //     count++;
    //   }
    // }
    // printf("\n");

    return native_scn_->output(0)->features().ptr<nvtype::half>();
  }

 private:
  SCNParameter param_;
  std::shared_ptr<Voxelization> voxelization_;//体素化
  std::shared_ptr<spconv::Engine> native_scn_;//自定义的引擎（load onnx之后，手动构建的engine
  unsigned int frame_ = 0;  // forward 调用帧计数，第 0 帧为 warmup 帧，不纳入 nsys 统计
};

std::shared_ptr<SCN> create_scn(const SCNParameter& param) {
  std::shared_ptr<SCNImplement> instance(new SCNImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace bevfusion