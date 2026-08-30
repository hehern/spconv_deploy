#ifndef __SPCONV_NODE_SPARSE_CONV_HPP__
#define __SPCONV_NODE_SPARSE_CONV_HPP__

#include <cuda_fp16.h>
#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

class SparseConvolution : public INode {
 public:
  SparseConvolution(const std::string& name, SparseDTensor* x,
                    const std::vector<int>& input_spatial_shape,
                    const std::vector<int>& output_spatial_shape,
                    const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape,
                    const std::vector<float>& weight_dynamic_ranges, const std::vector<unsigned short>& bias,
                    const std::vector<int>& bias_shape, const std::string& activation,
                    const std::vector<int>& kernel_size, const std::vector<int>& stride,
                    const std::vector<int>& padding, const std::vector<int>& dilation,
                    float input_dynamic_range, bool submanifold,
                    int max_output_points,const std::string& rulebook,
                    Precision precision, Precision output_precision,
                    const std::string& output_name);

  void forward(void *stream) override;

 private:
  unsigned short ndim = 3;
  unsigned short in_channels;
  unsigned short out_channels;
  // spconv参数
  nv::Tensor weight_;
  nv::Tensor weight2_;
  nv::Tensor bias_;
  std::vector<int> weight_shape_;
  std::vector<float> weight_dynamic_ranges_;
  std::vector<int> bias_shape_;
  std::string activation_;
  std::vector<int> kernel_size_;
  std::vector<int> stride_;
  std::vector<int> padding_;
  std::vector<int> dilation_;
  float input_dynamic_range_;
  bool submanifold_;
  int max_output_points_;
  std::string rulebook_;
  Precision precision_;//
  Precision output_precision_;
  std::vector<int> input_spatial_shape_;//eg:1440, 1440, 41
  std::vector<int> out_spatial_shape_;//eg:720, 720, 21
  nv::EventTimer timer_;
};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_SPARSE_CONV_HPP__