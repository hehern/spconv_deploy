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
 
#ifndef __SPCONV_ENGINE_HPP__
#define __SPCONV_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <algorithm>

#include "node.hpp"
#include "spconv/spconv_ops.h"
#include "common/timer.hpp"
#include "node_add.hpp"
#include "node_dense.hpp"
#include "node_relu.hpp"
#include "node_reshape.hpp"
#include "node_transpose.hpp"
#include "node_sparseconv.hpp"


namespace spconv {
/**
  Engine types for sparse convolution
**/
class Engine {
 public:
  /**
    Inference function for sparse convolution

    features_shape: The shape of the input feature matrix, it must be two elements.
    features_dtype: The data type of the input feature matrix, it must be Float16 now.
    features_data:  The data pointer of the input feature matrix
    indices_shape:  The shape of the input indices matrix, it must be two elements[n, 4]
    indices_dtype:  The data type of the input indices matrix, it must be Int32 now.
    indices_data:   The data pointer of the input indices matrix
    batch:          The batch size of the input, it must be 1 now.
    grid_size:      The grid size of the input data, For example: 41,1440,1440 or 1440,1440,41
    stream:         Which stream is expected to enqueue the inference.
  **/
  Engine(const std::vector<std::shared_ptr<SparseDTensor>>& input_tenosr, 
         const std::vector<std::shared_ptr<SparseDTensor>>& output_tenosr, 
         const std::vector<std::shared_ptr<INode>>& nodes):
         inputs_(input_tenosr),
         outputs_(output_tenosr),
         nodes_(nodes) {}
  void forward(const std::vector<int64_t>& features_shape,
               nv::DataType features_dtype, void* features_data,
               const std::vector<int64_t>& indices_shape, nv::DataType indices_dtype,
               void* indices_data, std::vector<int> grid_size, void *stream) {

    for(int i=0; i<nodes_.size(); i++) {
      nodes_[i]->set_is_computed(false);
    }
    SparseDTensor::clear_rulebooks();
    spconv::clear_indice_cache();
    inputs_[0]->set_data(features_shape, features_dtype, features_data, indices_shape, indices_dtype, indices_data, grid_size, stream);
    outputs_[0]->update(stream);
  }
  size_t num_input() const { return inputs_.size(); }
  size_t num_output() const {return outputs_.size(); }
  SparseDTensor* input(unsigned int index) { return inputs_[index].get(); }
  SparseDTensor* output(unsigned int index) { return outputs_[index].get(); }

 private:
  std::vector<std::shared_ptr<SparseDTensor>> inputs_;
  std::vector<std::shared_ptr<SparseDTensor>> outputs_;

  std::vector<std::shared_ptr<INode>> nodes_;
};

class EngineBuilder{
 public:
  void init() {
    inputs_.clear();
    outputs_.clear();
    nodes_.clear();
    engine_ = nullptr;
  }

  SparseDTensor* push_input(const std::string& name) {
    std::shared_ptr<SparseDTensor> x(new SparseDTensor(name));
    inputs_.push_back(x);
    return x.get();
  }

  INode* push_sparse_conv(const std::string& name, SparseDTensor* x,
                          const std::vector<int>& input_spatial_shape, const std::vector<int>& output_spatial_shape,
                          const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape,
                          const std::vector<float>& weight_dynamic_ranges, const std::vector<unsigned short>& bias,
                          const std::vector<int>& bias_shape, const std::string& activation,
                          const std::vector<int>& kernel_size, const std::vector<int>& stride,
                          const std::vector<int>& padding, const std::vector<int>& dilation,
                          float input_dynamic_range, bool submanifold,
                          int max_output_points,const std::string& rulebook,
                          Precision precision, Precision output_precision,
                          const std::string& output_name) {
    std::shared_ptr<INode> node(new SparseConvolution(name, x, input_spatial_shape, output_spatial_shape, weight, weight_shape, weight_dynamic_ranges, bias, bias_shape, activation,
    kernel_size, stride, padding, dilation, input_dynamic_range, submanifold, max_output_points, rulebook, precision, output_precision, output_name));
    nodes_.push_back(node);
    return node.get();
  }

  Add* push_add(const std::string& name, SparseDTensor* a, SparseDTensor* b, float a_dynamic_range, float b_dynamic_range,
                  const std::string& output_name, Precision precision, Precision output_precision) {
    std::shared_ptr<INode> node(new Add(name, a, b, a_dynamic_range, b_dynamic_range, output_name, precision, output_precision));
    nodes_.push_back(node);
    return static_cast<Add*>(node.get());
  }

  INode* push_relu(const std::string& name, SparseDTensor* x, const std::string& output_name) {
    std::shared_ptr<INode> node(new Relu(name, x, output_name));
    nodes_.push_back(node);
    return node.get();
  }

  INode* push_dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
                            const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) {
    std::shared_ptr<INode> node(new Dense(name, x, format, output_name, input_spatial_shape, output_shape));
    nodes_.push_back(node);
    return node.get();
  }

  INode* push_reshape(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& shape, const std::string& output_name) {
    std::shared_ptr<INode> node(new Reshape(name, x, shape, output_name));
    nodes_.push_back(node);
    return node.get();
  }

  INode* push_transpose(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& dims, const std::string& output_name) {
    std::shared_ptr<INode> node(new Transpose(name, x, dims, output_name));
    nodes_.push_back(node);
    return node.get();
  }

  void push_output(SparseDTensor* value) { 
    std::shared_ptr<SparseDTensor> sptr(value);
    outputs_.push_back(sptr);
  }

  std::shared_ptr<Engine> build(Precision precision, void* stream = nullptr) {
    engine_.reset(new Engine(inputs_, outputs_, nodes_));
    return engine_;
  }

 private:
  std::vector<std::shared_ptr<SparseDTensor>> inputs_;
  std::vector<std::shared_ptr<SparseDTensor>> outputs_;

  std::vector<std::shared_ptr<INode>> nodes_;
  std::shared_ptr<Engine> engine_ = nullptr;
};

/**
 * To build a engine.
*/
inline std::shared_ptr<EngineBuilder> create_engine_builder() {
  std::shared_ptr<EngineBuilder> instance(new EngineBuilder());
  instance->init();
  return instance;
}

/**
  Enable detailed information output

  enable: You should set this to true if you want to debug the model inference process. default:
  false
*/
inline void set_verbose(bool enable) {}
inline bool get_verbose() {}
inline const char* get_precision_string(Precision precision) {}



};  // namespace spconv

#endif  // #ifndef __SPCONV_ENGINE_HPP__