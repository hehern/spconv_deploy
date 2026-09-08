#include "node_add.hpp"
#include "normal_op/op_add.h"

namespace spconv {

Add::Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, float a_dynamic_range, float b_dynamic_range,
         const std::string& output_name, Precision precision, Precision output_precision) {
  input_.push_back(a);
  input_.push_back(b);

  output_ .push_back(new SparseDTensor(output_name, this));
  name_ = name;
}

void Add::forward(void *stream) {

  // step1:逐元素相加
  assert(input_[0]->features().shape[0] == input_[1]->features().shape[0]);
  assert(input_[0]->features().shape[1] == input_[1]->features().shape[1]);
  int64_t act_num = input_[0]->features().shape[0];
  int64_t voxel_dim = input_[0]->features().shape[1];
  
  nv::Tensor output_buffer = nv::Tensor::create(std::vector<int64_t>{act_num, voxel_dim}, input_[0]->features().dtype(), input_[0]->features().device());
  output_buffer.fill<half>(__float2half(0.0f), stream);//这里强制设置为half，后期做int8时候再改就好了
  // 这里参与add的两个输入中间差两个subm conv，所以不影响active voxel位置，直接加就行了
  add_cuda(input_[0]->features(), input_[1]->features(), output_buffer, act_num, voxel_dim, stream, relu_);

  // step2:调用输出的set_data将结果填充进去
  output_[0]->set_data(input_[0]->get_features_dtype(), output_buffer,
                       input_[0]->get_indices_dtype(), input_[0]->indices(),
                       input_[0]->grid_size());
  // std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv