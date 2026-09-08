#include "node_dense.hpp"
#include "normal_op/op_dense.h"

namespace spconv {

Dense::Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
             const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) {
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;

  input_spatial_shape_ = input_spatial_shape;//eg:180, 180, 2
  output_shape_ = output_shape;
  for(const auto& i : output_shape) {
    output_shape_64_.push_back(i);//eg:1, 128, 180, 180, 2
  }
}

void Dense::forward(void *stream) {
  // step1: 将稀疏张量转换为密集张量，shape为[1, features_shape[1], input_spatial_shape] ,eg:[1, 128, 180, 180, 2]
  int64_t act_num = input_[0]->features().shape[0];
  int64_t voxel_dim = input_[0]->features().shape[1];
  int64_t indices_dim = input_[0]->indices().shape[1];

  nv::Tensor output_buffer = nv::Tensor::create(output_shape_, nv::DataType::Float16);
  output_buffer.fill<half>(__float2half(0.0f), stream);
  dense_cuda(input_[0]->features(), input_[0]->indices(), output_buffer, act_num, voxel_dim, indices_dim, input_spatial_shape_, stream);

  // step2:
  output_[0]->set_data(input_[0]->get_features_dtype(), output_buffer,
                       input_[0]->get_indices_dtype(), input_[0]->indices(),
                       output_shape_);
  // std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv