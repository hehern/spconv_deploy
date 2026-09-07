#include "node_sparseconv.hpp"
#include "spconv/spconv_ops.h"

namespace spconv {

SparseConvolution::SparseConvolution(const std::string& name, SparseDTensor* x,
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
                                     const std::string& output_name) {
  // 输入输出变量初始化
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;

  // 参数初始化
  weight_shape_ = weight_shape;//[out_channel, kernel_size_x, kernel_size_y, kernel_size_z, in_channel]
  weight_dynamic_ranges_ = weight_dynamic_ranges;
  bias_shape_ = bias_shape;
  activation_ = activation;
  kernel_size_ = kernel_size;
  stride_ = stride;
  padding_ = padding;
  dilation_ = dilation;
  input_dynamic_range_ = input_dynamic_range;
  submanifold_ = submanifold;
  max_output_points_ = max_output_points;
  rulebook_ = rulebook;
  precision_ = precision;
  output_precision_ = output_precision;
  input_spatial_shape_ = input_spatial_shape;//eg:1440, 1440, 41
  out_spatial_shape_ = output_spatial_shape;//eg:720, 720, 21

  // weight转换为[kernel_size_x*kernel_size_y*kernel_size_z, in_channel, out_channel]格式
  int kernel_x = weight_shape_[1], kernel_y = weight_shape_[2], kernel_z = weight_shape_[3];
  int in_channel = weight_shape_[4];
  int out_channel = weight_shape_[0];
  int new_size = kernel_x * kernel_y * kernel_z * in_channel * out_channel;
  std::vector<unsigned short> result(new_size);

  int index_in_flattened = 0;  
  for (int oc = 0; oc < out_channel; ++oc) {
    for (int kx = 0; kx < kernel_x; ++kx) {
      for (int ky = 0; ky < kernel_y; ++ky) {
        for (int kz = 0; kz < kernel_z; ++kz) {
          for (int ic = 0; ic < in_channel; ++ic) {
            int kernel_index = kz + ky * kernel_z + kx * kernel_y * kernel_z;
            int index = kernel_index * in_channel * out_channel + ic * out_channel + oc;
            result[index] = weight[index_in_flattened++];
          }  
        }  
      }  
    }  
  }
  weight_ = nv::Tensor::from_data(&result[0], std::vector<int64_t>{kernel_x*kernel_y*kernel_z, in_channel, out_channel}, nv::DataType::Float16, false);
  weight2_ = nv::Tensor::from_data(&weight[0], std::vector<int64_t>{out_channel, kernel_x*kernel_y*kernel_z, in_channel}, nv::DataType::Float16, false);
  bias_ = nv::Tensor::from_data(&bias[0], bias_shape, nv::DataType::Float16, false);
  weight_.to_device_();
  weight2_.to_device_();
  bias_.to_device_();
}

void SparseConvolution::forward(void *stream) {
  // std::cout << name_ << " forward:" << std::endl;
  // if (name_ == "conv0") {
  //   printFeatures(input_[0]->features(), stream);
  // }
  // if (name_ == "conv0") {
  //   printFeatures(weight_, stream);
  // }
  // step1:查找/计算rulebook
  std::vector<nv::Tensor> datas = SparseDTensor::find_indice_pair(rulebook_);
  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  if (datas.empty()) {
    // std::cout << "no rulebook" << std::endl;
    // timer_.start(_stream);
    // datas = getIndicePairs(input_[0]->indices(), out_spatial_shape_, input_spatial_shape_, kernel_size_, stride_, padding_, dilation_, submanifold_, stream);
    datas = getIndicePairsImplicitGemm(input_[0]->indices(), out_spatial_shape_, input_spatial_shape_, kernel_size_, stride_, padding_, dilation_, submanifold_, stream);
    SparseDTensor::add_rulebook(rulebook_, datas);
    // timer_.stop("getIndicePairs done");
    // std::cout << "add rulebook done" << std::endl;
  }

  // step2:conv计算 (bias + ReLU 已融合进 conv_kernel 的 epilogue:
  // ConvParams d_is_bias=true -> ConstOutIterator 按 K 维广播读 bias, beta=1, act=ReLU)
  nv::Tensor result = implicit_gemm(input_[0]->features(), weight2_, datas[1], datas[2], datas[3],
                                    datas[4].to_host().at<int32_t>(0), submanifold_,
                                    bias_, activation_ == "ReLU", stream);
  // addBiasAndRelu(result, bias_, activation_=="ReLU", stream);

  // judgeIndicesOutshape(datas[0], out_spatial_shape_, stream);
  // if (name_ == "conv0") {
  //   printFeatures(result, stream);
  // }

  // step3:保存输出
  // std::vector<int64_t> features_shape{datas[0].shape[0], weight_shape_[0]};
  // std::vector<int64_t> indices_shape{datas[0].shape[0], datas[0].shape[1]};
  // std::cout << "features_shape = " << datas[0].shape[0] << "," << weight_shape_[0] << std::endl;
  // std::cout << "indices_shape = " << datas[0].shape[0] << "," << datas[0].shape[1] << std::endl;
  // cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  // timer_.start(_stream);
  // printf("result use_conut: %d\n", result.use_count());
  output_[0]->set_data(input_[0]->get_features_dtype(), result,
                       input_[0]->get_indices_dtype(), datas[0],
                       out_spatial_shape_);
  // printf("result use_conut: %d\n", result.use_count());
  // timer_.stop("SparseConvolution set_data");
}

}// namespace spconv