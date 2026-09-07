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

#include "bevfusion.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"

namespace bevfusion {

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    camera_bevpool_ =
        camera::create_bevpool(camera_backbone_->camera_shape(), param.geometry.geometry_dim.x, param.geometry.geometry_dim.y);
    if (camera_bevpool_ == nullptr) {
      printf("Failed to create camera bevpool.\n");
      return false;
    }

    camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    if (camera_vtransform_ == nullptr) {
      printf("Failed to create camera vtransform.\n");
      return false;
    }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    transbbox_ = head::transbbox::create_transbbox(param.transbbox);
    if (transbbox_ == nullptr) {
      printf("Failed to create head transbbox.\n");
      return false;
    }

    lidar_scn_ = lidar::create_scn(param.lidar_scn);//SCNImplement
    if (lidar_scn_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    camera_depth_ = camera::create_depth(param.normalize.output_width, param.normalize.output_height, param.normalize.num_camera);
    if (camera_depth_ == nullptr) {
      printf("Failed to create depth.\n");
      return false;
    }

    camera_geometry_ = camera::create_geometry(param.geometry);
    if (camera_geometry_ == nullptr) {
      printf("Failed to create geometry.\n");
      return false;
    }

    capacity_points_ = 300000;
    bytes_capacity_points_ = capacity_points_ * param.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));

    // 预填 tensor 内存池 (通用阶梯约 260MB): 之后 Lidar/相机热路径的 tensor 创建
    // 全部 best-fit 命中预填块, 消除第一帧的池冷启动 cudaMalloc
    // (kernel 模块加载等一次性开销仍由调用方的 warmup 帧覆盖)。
    nv::Tensor::pool_prime();

    param_ = param;
    return true;
  }

  std::vector<head::transbbox::BoundingBox> forward_timer(const void* camera_images, const nvtype::half* lidar_points,
                                                          int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    timer_.start(_stream);

    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));//将当前帧激光雷达点云拷贝到gpu上-lidar_points_device_
    timer_.stop("[NoSt] CopyLidar");//timer_.start与stop配合计时

    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);//图片resize crop normal
      timer_.stop("[NoSt] ImageNrom");
    }

    timer_.start(_stream);
    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);//体素化以及sparseencoder
    times.emplace_back(timer_.stop("Lidar Backbone"));

    timer_.start(_stream);
    const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);//将激光点云投影到图片上，获得深度真值
    times.emplace_back(timer_.stop("Camera Depth"));

    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, depth, stream);
    times.emplace_back(timer_.stop("Camera Backbone"));

    timer_.start(_stream);
    const nvtype::half* camera_bev = this->camera_bevpool_->forward(
        this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
        this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
    times.emplace_back(timer_.stop("Camera Bevpool"));

    timer_.start(_stream);
    const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    times.emplace_back(timer_.stop("VTransform"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    times.emplace_back(timer_.stop("Transfusion"));

    timer_.start(_stream);
    auto output =
        this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream, param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("Head BoundingBox"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return output;
  }

  virtual std::vector<head::transbbox::BoundingBox> forward(const unsigned char** camera_images, const nvtype::half* lidar_points,
                                                            int num_points, void* stream) override {
    return this->forward_timer(camera_images, lidar_points, num_points, stream, true);
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }

  virtual void print() override {
    camera_backbone_->print();
    camera_vtransform_->print();
    transfusion_->print();
    transbbox_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream) override {
    camera_depth_->update(img_aug_matrix, lidar2image, stream);
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  }

  virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Depth> camera_depth_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<lidar::SCN> lidar_scn_;//lidar sparse conv net
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<head::transbbox::TransBBox> transbbox_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion