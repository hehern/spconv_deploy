#include "node_transpose.hpp"
#include "normal_op/op_transpose.h"

namespace spconv {

Transpose::Transpose(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& dims, const std::string& output_name) {
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;

  dims_ = dims;//0, 1, 4, 2, 3
}

void Transpose::forward(void *stream) {
  // std::cout << name_ << ", forward begin:" << std::endl;
  assert(input_[0]->grid_size().size() == dims_.size());
  // step1:转换数据保存的顺序[1, 128, 180, 180, 2]->[1, 128, 2, 180, 180]
  std::vector<int> input_shape = input_[0]->grid_size();
  output_shape_.resize(input_shape.size());
  output_shape_64_.resize(input_shape.size());
  for (int i=0; i<input_shape.size(); i++) {
    output_shape_[i] = input_shape[dims_[i]];
    output_shape_64_[i] = input_shape[dims_[i]];
  }
  // for(auto i : output_shape_) {
  //   std::cout << i << ", ";
  // }
  // std::cout << std::endl;

  int64_t act_num = input_[0]->indices().shape[0];
  int64_t voxel_dim = input_[0]->features().shape[1];//128
  int64_t indices_dim = input_[0]->indices().shape[1];
  nv::Tensor output_buffer = nv::Tensor::create(output_shape_, nv::DataType::Float16);
  output_buffer.fill<half>(__float2half(0.0f), stream);
  // transpose_cuda(input_[0]->features(), input_[0]->indices(), output_buffer, act_num, voxel_dim, indices_dim, input_shape, output_shape_, stream);//注意这里只需要转换那些有效voxel就可以了
  transpose_with_cuda(input_[0]->features(), output_buffer, input_shape, stream);
  // step2:填充数据
  output_[0]->set_data(input_[0]->get_features_dtype(), output_buffer,
                       input_[0]->get_indices_dtype(), input_[0]->indices(),
                       output_shape_);
  // std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv