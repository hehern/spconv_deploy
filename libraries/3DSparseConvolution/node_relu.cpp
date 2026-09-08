#include "node_relu.hpp"
#include "normal_op/op_relu.h"

namespace spconv {

Relu::Relu(const std::string& name, SparseDTensor* x, const std::string& output_name) {
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;
}

void Relu::forward(void *stream) {
  // step1:relu
  int64_t act_num = input_[0]->features().shape[0];
  int64_t voxel_dim = input_[0]->features().shape[1];

  nv::Tensor output_buffer = nv::Tensor::create(std::vector<int64_t>{act_num, voxel_dim}, input_[0]->features().dtype(), input_[0]->features().device());
  output_buffer.fill<half>(__float2half(0.0f), stream);
  relu_cuda(input_[0]->features(), output_buffer, act_num, voxel_dim, stream);

  // step2:调用输出的set_data将结果填充进去
  output_[0]->set_data(input_[0]->get_features_dtype(), output_buffer,
                       input_[0]->get_indices_dtype(), input_[0]->indices(),
                       input_[0]->grid_size());
  // std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv