#ifndef CONVOUTLOC_H_
#define CONVOUTLOC_H_

#include <array>

#ifdef __CUDACC__
#define TV_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define TV_HOST_DEVICE_INLINE inline
#endif
namespace spconv {

struct ConvOutLocIter {
  int kernelSize_[3];
  int stride_[3];
  int padding_[3];
  int dilation_[3];
  int kernel_offset_;
  int count_[3];
  int output_dims_[3];
  int input_dims_[3];

  TV_HOST_DEVICE_INLINE ConvOutLocIter(
      const int* kernel_size, const int* stride, 
      const int* padding, const int* dilation,
      const int* output_dims, const int* input_dims) :
      kernelSize_{kernel_size[0], kernel_size[1], kernel_size[2]},
      stride_{stride[0], stride[1], stride[2]},
      padding_{padding[0], padding[1], padding[2]},
      dilation_{dilation[0], dilation[1], dilation[2]},
      kernel_offset_(0), count_{0, 0, 0}, 
      output_dims_{output_dims[0], output_dims[1], output_dims[2]},
      input_dims_{input_dims[0], input_dims[1], input_dims[2]} {
  }
  TV_HOST_DEVICE_INLINE void set_filter_offset(int kernel_offset) {
    kernel_offset_ = kernel_offset;
    int residual = kernel_offset;
    count_[0] = int(residual / kernelSize_[0]);
    residual = residual % kernelSize_[0];
    count_[1] = int(residual / kernelSize_[1]);
    count_[2] = int(residual % kernelSize_[1]);
  }
  template <bool NoStride>
  TV_HOST_DEVICE_INLINE void nhw_to_npq(const int* nhw_offset, int* out)  const {
    
    int r_0 = count_[0];
    int h_0 = (nhw_offset[1] + padding_[0] - 
        r_0 * dilation_[0]) / (NoStride ? 1 : stride_[0]);
    int r_1 = count_[1];
    int h_1 = (nhw_offset[2] + padding_[1] - 
        r_1 * dilation_[1]) / (NoStride ? 1 : stride_[1]);
    int r_2 = count_[2];
    int h_2 = (nhw_offset[3] + padding_[2] - 
        r_2 * dilation_[2]) / (NoStride ? 1 : stride_[2]);
    out[0] = nhw_offset[0];
    out[1] = h_0;
    out[2] = h_1;
    out[3] = h_2;
  }
  TV_HOST_DEVICE_INLINE void npq_to_nhw(const int* npq_offset, int* out)  const {
    
    int r_0 = count_[0];
    int h_0 = npq_offset[1] * stride_[0] - padding_[0] + r_0 * dilation_[0];
    int r_1 = count_[1];
    int h_1 = npq_offset[2] * stride_[1] - padding_[1] + r_1 * dilation_[1];
    int r_2 = count_[2];
    int h_2 = npq_offset[3] * stride_[2] - padding_[2] + r_2 * dilation_[2];
    out[0] = npq_offset[0];
    out[1] = h_0;
    out[2] = h_1;
    out[3] = h_2;
  }
  TV_HOST_DEVICE_INLINE bool query_npq(const int* nhw_offset, int (&npq_offset)[4])  const {
    
    int npq_no_stride[4];
    nhw_to_npq<true>(nhw_offset, npq_no_stride);
    npq_offset[0] = npq_no_stride[0];
    npq_offset[1] = npq_no_stride[1] / stride_[0];
    npq_offset[2] = npq_no_stride[2] / stride_[1];
    npq_offset[3] = npq_no_stride[3] / stride_[2];
    return npq_offset[1] >= 0 && npq_offset[1] < output_dims_[0] && npq_offset[2] >= 0 && npq_offset[2] < output_dims_[1] && npq_offset[3] >= 0 && npq_offset[3] < output_dims_[2] &&
        !(npq_no_stride[1] % stride_[0]) && !(npq_no_stride[2] % stride_[1]) && !(npq_no_stride[3] % stride_[2]);
  }
  TV_HOST_DEVICE_INLINE bool query_nhw(const int* npq_offset, int (&nhw_offset)[4])  const {
    
    npq_to_nhw(npq_offset, nhw_offset);
    return nhw_offset[1] >= 0 && nhw_offset[1] < input_dims_[0] && nhw_offset[2] >= 0 && nhw_offset[2] < input_dims_[1] && nhw_offset[3] >= 0 && nhw_offset[3] < input_dims_[2];
  }
  TV_HOST_DEVICE_INLINE int32_t layout_npq(const int* indexes)  const {
    
    return (indexes[1] * output_dims_[1] + indexes[2]) * output_dims_[2] + indexes[3];
  }
  TV_HOST_DEVICE_INLINE void inverse(int32_t index, int* out)  const {

    int32_t residual = index;
    out[3] = int(residual % output_dims_[2]);
    residual -= out[3];
    residual /= output_dims_[2];
    out[2] = int(residual % output_dims_[1]);
    residual -= out[2];
    residual /= output_dims_[1];
    out[1] = int(residual % output_dims_[0]);


  }
};
} // namespace spconv

#endif