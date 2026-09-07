#include <limits>
#include <iostream>
#include <algorithm>
#include <map>
#include <utility>
#include <string>
#include <numeric>


#include <cuda_fp16.h>

#include "spconv_ops.h"
#include "conv/ConvOutLocIter.h"
#include "common/check.hpp"
#include "common/timer.hpp"
namespace spconv {

// indiceConv2 的 static 缓存
// buffer缓存不清除(cudaFree开销极大), offsets缓存随rulebook失效需清除
static std::map<std::pair<int64_t, int64_t>, nv::Tensor> g_inputBuffer_cache;
static std::map<std::pair<int64_t, int64_t>, nv::Tensor> g_outputBuffer_cache;
static std::map<std::string, std::pair<nv::Tensor, nv::Tensor>> g_offsets_kernels_cache;

void clear_indice_cache() {
  // g_inputBuffer_cache.clear(); //这两个暂时不clear，因为比较耗时，但是随着代码运行会不停因为active voxel个数不一样，会逐渐push
  // g_outputBuffer_cache.clear();
  g_offsets_kernels_cache.clear();
}

/*
  in:
  indices:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  outSpatialShape:vector<int>, size()==3, 输出体素栅格shape,eg:{720, 720, 21},xyz的顺序
  spatialShape:vector<int>, size()==3, 输入体素栅格shape,eg:{1440, 1440, 41}
  kernelSize:vector<int>, size()==3,eg:{3, 3, 3}
  stride:vector<int>, size()==3,eg:{1, 1, 1}
  padding:vector<int>, size()==3,eg:{1, 1, 1}
  dilation:vector<int>, size()==3,eg:{1, 1, 1}
  out:
  indices: nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  indicePairs: shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout
  indiceNum: nv::Tensor, shape:{27},对应的是rule_book中的count
*/
std::vector<nv::Tensor>
getIndicePairs(nv::Tensor indices,
               std::vector<int> outSpatialShape,
               std::vector<int> spatialShape,
               std::vector<int> kernelSize, std::vector<int> stride,
               std::vector<int> padding, std::vector<int> dilation,
               bool subM, void* stream) {
  int64_t NDim = kernelSize.size();//3
  auto numAct = indices.shape[0];//type:long int
  auto coorDim = indices.shape[1] - 1; // batchIdx + xyz
  Asserts(NDim == coorDim, "error");
  Asserts(int64_t(kernelSize.size()) == coorDim, "error");
  Asserts(int64_t(outSpatialShape.size()) == coorDim, "error");
  Asserts(int64_t(stride.size()) == coorDim, "error");
  Asserts(int64_t(padding.size()) == coorDim, "error");
  Asserts(int64_t(dilation.size()) == coorDim, "error");
  int64_t kernelVolume = kernelSize[0];
  for (size_t i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }//27
  Asserts(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (size_t i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }//720*720*21=10886400
  std::string msg = "due to limits of cuda hash, the volume of dense space "
                    "include batch size ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  if (!(outputVolume < std::numeric_limits<int64_t>::max())) {
    fprintf(stderr, "Assert failed. outputVolume check in file %s:%d, message: %s\n", __FILE__, __LINE__, msg.c_str());
    abort();
  }
  nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{2, kernelVolume, numAct}, nv::DataType::Int32);//shape:{2,27,n}
  // indicePairs.fill<int32_t>(-1);
  indicePairs.memset(0xFF, stream);
  nv::Tensor indiceNum = nv::Tensor::create(std::vector<int64_t>{kernelVolume}, nv::DataType::Int32);//shape:{27}
  // indiceNum.fill<int32_t>(0);
  indiceNum.memset(0, stream);
  nv::Tensor gridOut = nv::Tensor::create(std::vector<int64_t>{outputVolume}, nv::DataType::Int32);//输出tensor，展平为1维的
  // gridOut.fill<int32_t>(-1);
  gridOut.memset(0xFF, stream);

  nv::Tensor ou = nv::Tensor::create(std::vector<int64_t>{NDim}, nv::DataType::Int32);//output_shape
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), outSpatialShape.size()*sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t)stream));

  // 参考资料：https://zhuanlan.zhihu.com/p/383299678
  int64_t numActOut = -1;//如果subM类型的spconv，输出actnum和输入actnum是一致的，如果subM为false，则需要计算
  nv::EventTimer timer_;
  // cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  if (subM) {//子流行卷积
    // timer_.start(_stream);
    numActOut = create_submconv_indice_pair_cuda(indices, gridOut, indicePairs, indiceNum, ou, outputVolume, stream);
    // timer_.stop("create_submconv_indice_pair_cuda");
    return {indices, indicePairs, indiceNum};
  } else {//非子流行卷积
    // std::cout << "not subm" << std::endl;
    // checkRuntime(cudaStreamSynchronize(_stream));
    nv::Tensor indicePairUnique = nv::Tensor::create(std::vector<int64_t>{int64_t(indicePairs.numel / 2) + 1}, nv::DataType::Int32);//N*2*27/2+1
    indicePairUnique.fill<int32_t>(std::numeric_limits<int32_t>::max());
    nv::Tensor outInds = nv::Tensor::create(std::vector<int64_t>{numAct * kernelVolume, coorDim + 1}, nv::DataType::Int32);//{n*27, 4}，这里定义numAct * kernelVolume是因为非子流行卷积输出active voxel个数比输入多，所以这里相当于设置了一个极限最大值
    // outInds.fill<int32_t>(0);
    outInds.memset(0, stream);
    // checkRuntime(cudaStreamSynchronize(_stream));
    // timer_.start(_stream);
    numActOut = create_conv_indice_pair_p1_cuda(indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride, padding, dilation, outSpatialShape, outputVolume, stream);
    // timer_.stop("create_conv_indice_pair_p1_cuda");
    // std::cout << "not subm, rulebook 1, numActOut = " << numActOut << std::endl;
    if (numActOut > 0) {
      // timer_.start(_stream);
      nv::Tensor indicePairUnique_new = find_unique_elements_cuda(indicePairUnique, stream);//挑出tensor中的独立不重复元素,并按照升序排列，indicePairUnique中保存的是vout即输出voxel grid的一维index
      // timer_.stop("find_unique_elements_cuda");
      // std::cout << "not subm, rulebook 2, find_unique_elements_cuda done" << std::endl;
      // timer_.start(_stream);
      numActOut = create_conv_indice_pair_p2_cuda(indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique_new, outSpatialShape, stream);
      // timer_.stop("create_conv_indice_pair_p2_cuda");
      // std::cout << "not subm, rulebook 2, numActOut = " << numActOut << std::endl;
    }
    // timer_.start(_stream);
    nv::Tensor finalOutInds = nv::Tensor::from_data(outInds.ptr<int>(), std::vector<int64_t>{numActOut, coorDim + 1}, nv::DataType::Int32);//切片，这地方用from_data有点浪费了，可以优化
    // timer_.stop("SparseConvolution set_data");
    return {finalOutInds, indicePairs, indiceNum};
  }
}

nv::Tensor indiceConv(nv::Tensor features,    // 输入特征(N,inchannel)
                      nv::Tensor filters,     // eg:权重[3*3*3,5,16],5为输入channel个数，16为输出channel个数
                      nv::Tensor indicePairs, // [2, 27, N]
                      nv::Tensor indiceNum,   // [27]，用于保存卷积核每一个位置上的总的计算的次数
                      int64_t numActOut,      // 输出有效voxel个数，暂时可以用M表示
                      bool subM,              // 子流线卷积默认 true
                      void* stream) {

  auto kernelVolume = indiceNum.size(0); // 27
  auto numInPlanes = features.size(1);   // 5
  auto numOutPlanes = filters.size(2);   // 16
  auto numActIn = indicePairs.size(2);

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  nv::EventTimer timer_;

  auto indicePairNumCpu = indiceNum.to_host();
  // timer_.start(_stream);
  nv::Tensor output = nv::Tensor::create(std::vector<int64_t>{numActOut, numOutPlanes}, features.dtype(), features.device());
  output.fill<half>(__float2half(0.0f));
  // output.memset(0, stream);
  // timer_.stop("tensor fill");

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2; // 13
  int indicePairMaxSize = numActOut;          // N
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // timer_.start(_stream);
    matrix_multiply_cuda(features, filters, output, numActOut, numOutPlanes, numInPlanes, indicePairMaxOffset, stream);
    // timer_.stop("matrix_multiply_cuda");

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {//subm情况下，最大的indicePairSize肯定是kernel中心点，现在计算的是第二大，如果第二大为0的话，表示每次conv没有非kernel中心点参与卷积，直接返回计算就结束了
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + kernelVolume);
  }

  // 原版本：逐kernel循环
  // 按照rulebook逐卷积核元素计算
  nv::Tensor inputBuffer = nv::Tensor::create(std::vector<int64_t>{indicePairMaxSize, numInPlanes}, features.dtype(), features.device());
  nv::Tensor outputBuffer = nv::Tensor::create(std::vector<int64_t>{indicePairMaxSize, numOutPlanes}, features.dtype(), features.device());

  for (int i = 0; i < kernelVolume; ++i) {//27
    auto nHot = indicePairNumCpu.ptr<int>()[i];//表示第i个卷积核元素对应激活输入输出对个数(count)
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {//count为0,或者subm情况下卷积核中心位置情况下，不需要计算直接continue
      continue;
    }

    // timer_.start(_stream);
    sparse_gather_cuda(inputBuffer, features, indicePairs, nHot, i*numActIn, stream);//根据indicePairs中的vin查找到对应的输入voxels的值，并保存在inputBuffer
    // timer_.stop("sparse_gather_cuda");
    // timer_.start(_stream);
    matrix_multiply_cuda(inputBuffer, filters, outputBuffer, nHot, numOutPlanes, numInPlanes, i, stream);//gemm
    // timer_.stop("matrix_multiply_cuda");
    // timer_.start(_stream);
    sparse_scatter_add_cuda(outputBuffer, output, indicePairs, nHot, (kernelVolume+i)*numActIn, stream);//将结果填充到output中去
    // timer_.stop("sparse_scatter_add_cuda");
  }

  return output;
}

nv::Tensor indiceConv2(nv::Tensor features,    // 输入特征(N,inchannel)
                       nv::Tensor filters,     // eg:权重[3*3*3,5,16],5为输入channel个数，16为输出channel个数
                       nv::Tensor indicePairs, // [2, 27, N]
                       nv::Tensor indiceNum,   // [27]，用于保存卷积核每一个位置上的总的计算的次数
                       int64_t numActOut,      // 输出有效voxel个数，暂时可以用M表示
                       bool subM,              // 子流线卷积默认 true
                       const std::string &rulebook,
                       void* stream) {

  auto kernelVolume = indiceNum.size(0); // 27
  auto numInPlanes = features.size(1);   // 5
  auto numOutPlanes = filters.size(2);   // 16
  auto numActIn = indicePairs.size(2);

  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  nv::EventTimer timer_;

  auto indicePairNumCpu = indiceNum.to_host();

  nv::Tensor output = nv::Tensor::create(std::vector<int64_t>{numActOut, numOutPlanes}, features.dtype(), features.device());
  output.memset(0, stream);//其实清不清零都无所谓

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2; // 13
  int indicePairMaxSize = numActOut;          // N
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // timer_.start(_stream);
    matrix_multiply_cuda(features, filters, output, numActOut, numOutPlanes, numInPlanes, indicePairMaxOffset, stream);
    // timer_.stop("matrix_multiply_cuda");

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {//subm情况下，最大的indicePairSize肯定是kernel中心点，现在计算的是第二大，如果第二大为0的话，表示每次conv没有非kernel中心点参与卷积，直接返回计算就结束了
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.ptr<int>(),
                        indicePairNumCpu.ptr<int>() + kernelVolume);
  }

  // 计算总输入/输出数量
  int totalCount = 0;
  int subMCenterKernel = subM ? indicePairMaxOffset : -1;
  const int* indiceNumPtr = indicePairNumCpu.ptr<int>();
  for (int i = 0; i < kernelVolume; ++i) {
    if (i != subMCenterKernel) {
      totalCount += indiceNumPtr[i];
    }
  }

  if (totalCount <= 0) {
    return output;
  }

  // 一次性gather所有kernel位置的输入到连续buffer
  // static缓存: rulebook相同则totalCount相同, 复用避免反复cudaMalloc
  auto& inputBuffer_cache  = g_inputBuffer_cache;
  auto& outputBuffer_cache = g_outputBuffer_cache;

  nv::Tensor allInputBuffer;
  auto inKey = std::make_pair(totalCount, numInPlanes);
  auto inIt = inputBuffer_cache.find(inKey);
  if (inIt == inputBuffer_cache.end()) {
    allInputBuffer = nv::Tensor::create(
        std::vector<int64_t>{totalCount, numInPlanes}, features.dtype(), features.device());
    inputBuffer_cache[inKey] = allInputBuffer;
  } else {
    allInputBuffer = inIt->second;
  }

  nv::Tensor allOutputBuffer;
  auto outKey = std::make_pair(totalCount, numOutPlanes);
  auto outIt = outputBuffer_cache.find(outKey);
  if (outIt == outputBuffer_cache.end()) {
    allOutputBuffer = nv::Tensor::create(
        std::vector<int64_t>{totalCount, numOutPlanes}, features.dtype(), features.device());
    outputBuffer_cache[outKey] = allOutputBuffer;
  } else {
    allOutputBuffer = outIt->second;
  }

  // 预计算每个kernel位置的偏移量和kernel标识数组
  // static缓存: rulebook相同则数据相同, 避免反复计算和host→device拷贝
  auto& offsets_kernels_cache = g_offsets_kernels_cache;
  nv::Tensor offsetsTensor, kernelIdsTensor;
  auto it = offsets_kernels_cache.find(rulebook);
  if (it != offsets_kernels_cache.end()) {
    offsetsTensor  = it->second.first;
    kernelIdsTensor = it->second.second;
  } else {
    std::vector<int> offsets_host(kernelVolume + 1, 0);
    std::vector<int> kernelIds_host;
    kernelIds_host.reserve(totalCount);

    for (int i = 0; i < kernelVolume; ++i) {
      int cnt = (i != subMCenterKernel ? indiceNumPtr[i] : 0);
      offsets_host[i + 1] = offsets_host[i] + cnt;
      kernelIds_host.insert(kernelIds_host.end(), cnt, i);
    }

    offsetsTensor = nv::Tensor::create(std::vector<int64_t>{kernelVolume + 1}, nv::DataType::Int32, true);
    offsetsTensor.copy_from_host(offsets_host.data(), stream);

    kernelIdsTensor = nv::Tensor::create(std::vector<int64_t>{totalCount}, nv::DataType::Int32, true);
    kernelIdsTensor.copy_from_host(kernelIds_host.data(), stream);

    offsets_kernels_cache[rulebook] = {offsetsTensor, kernelIdsTensor};
  }

  // 一次性gather
  sparse_gather_all_cuda(allInputBuffer, features, indicePairs, kernelIdsTensor.ptr<int>(),
                         offsetsTensor.ptr<int>(), numActIn, totalCount, stream, kernelVolume);

  // 按kernel位置循环执行GEMM
  half* inputBase = allInputBuffer.ptr<half>();
  half* outputBase = allOutputBuffer.ptr<half>();
  int currentOffset = 0;

  for (int i = 0; i < kernelVolume; ++i) {
    if (i == subMCenterKernel) continue;
    auto nHot = indiceNumPtr[i];
    if (nHot <= 0) continue;

    matrix_multiply_cuda(inputBase + currentOffset * numInPlanes, filters, 
                         outputBase + currentOffset * numOutPlanes,
                         nHot, numOutPlanes, numInPlanes, i, stream);

    currentOffset += nHot;
  }

  // 一次性scatter所有结果回output
  sparse_scatter_add_all_cuda(allOutputBuffer, output, indicePairs, kernelIdsTensor.ptr<int>(),
                              offsetsTensor.ptr<int>(), numActIn, totalCount, stream, kernelVolume);

  return output;
}

void printFeatures(nv::Tensor features, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  checkRuntime(cudaStreamSynchronize(_stream));

  auto featuresHost = features.to_host(stream);
  auto f_h_ptr = featuresHost.ptr<half>();
  printf("features numel = %d\n", featuresHost.numel);
  for(size_t i=0; i<featuresHost.numel; i++) {
    float f_f = __half2float(f_h_ptr[i]);
    printf("%f,", f_f);
    if ((i+1)%features.shape[1] == 0) {
      printf("\n");
    }

  }
  printf("\n");
}

int get_handcrafted_max_act_out(size_t num_act_in, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation)   {
  
  int res = num_act_in;
  for (int i = 0; i < ksize.size(); ++i){
    if (ksize[i] <= stride[i]){
      res *= 1;
    } else if (ksize[i] > stride[i]){
      res *= divup(ksize[i], stride[i]);
    } else{
      res *= ksize[i];
    }
  }
  return res;
}


std::vector<nv::Tensor>
getIndicePairsImplicitGemm(nv::Tensor indices,
                           std::vector<int> outSpatialShape,
                           std::vector<int> spatialShape,
                           std::vector<int> kernelSize,
                           std::vector<int> stride,
                           std::vector<int> padding,
                           std::vector<int> dilation,
                           bool subm,
                           void* stream)   { 

  int kernelVolume = std::accumulate(kernelSize.begin(), kernelSize.end(), 1, std::multiplies<int>());//kernel volume,eg:27
  ConvOutLocIter loc_iter(kernelSize.data(), stride.data(), padding.data(), dilation.data(), outSpatialShape.data(), spatialShape.data());
  int64_t numAct = indices.shape[0];

  int num_act_out = 0;
  if (subm) {
    num_act_out = numAct;

    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, numAct}, nv::DataType::Int32);
    indicePairs.fill<int32_t>(-1);
    
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::UInt32);//每个active voxel都与kernel中的哪个元素进行卷积的mask
    generate_subm_conv_inds(indices, hash_k, hash_v, indicePairs,
        spatialShape, kernelSize, pair_mask, loc_iter, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{numAct}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, stream);//对pair_mask进行排序，返回排序后的索引

    nv::Tensor numActOut = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, false);
    numActOut.ptr<int32_t>()[0] = num_act_out;
    
    return {indices, indicePairs, pair_mask, mask_argsort, numActOut};

  } else {
    auto pair_size = kernelVolume * numAct;
    nv::Tensor indice_pairs_uniq = nv::Tensor::create(std::vector<int64_t>{pair_size + 1}, nv::DataType::Int32);

    generate_conv_inds_mask_stage1(indices, indice_pairs_uniq, kernelSize, loc_iter, stream);
    nv::Tensor indicePairUnique_new = find_unique_elements_cuda(indice_pairs_uniq, stream);//挑出tensor中的独立不重复元素,并按照升序排列，indicePairUnique中保存的是vout即输出voxel grid的一维index
    num_act_out = indicePairUnique_new.shape[0];

    nv::Tensor hash_k = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    nv::Tensor hash_v = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    hash_k.fill<int32_t>(std::numeric_limits<int32_t>::max());
    nv::Tensor out_inds = nv::Tensor::create(std::vector<int64_t>{num_act_out, indices.shape[1]}, indices.dtype());
    nv::Tensor indicePairs = nv::Tensor::create(std::vector<int64_t>{kernelVolume, num_act_out}, indices.dtype());
    indicePairs.fill<int32_t>(-1);
    nv::Tensor pair_mask = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::UInt32);
    pair_mask.fill<uint32_t>(0);

    generate_conv_inds_mask_stage2(indices, hash_k, hash_v, indicePairs,
        indicePairUnique_new, indice_pairs_uniq,
        out_inds, pair_mask, num_act_out, kernelSize, loc_iter, stream);

    nv::Tensor mask_argsort = nv::Tensor::create(std::vector<int64_t>{num_act_out}, nv::DataType::Int32);
    sort_1d_by_key_allocator_v2(pair_mask, mask_argsort, stream);
    nv::Tensor numActOut = nv::Tensor::create(std::vector<int64_t>{1}, nv::DataType::Int32, false);
    numActOut.ptr<int32_t>()[0] = num_act_out;
    return {out_inds, indicePairs, pair_mask, mask_argsort, numActOut};
  
  }
}

/*
  indicePairs: shape:{27,n},就是rule_book，存储参与当前kernel位置卷积的active voxel的序号[0, numActOut-1]
  pair_mask: shape:{n},每个active voxel都与kernel中的哪个元素进行卷积的mask
  mask_argsort: shape:{n},对pair_mask进行排序，返回排序后的索引
*/
nv::Tensor
implicit_gemm(nv::Tensor features,
              nv::Tensor filters, //格式为eg:权重[16,3*3*3,5]，输出channel kernel_volume 输入channel
              nv::Tensor indicePairs,
              nv::Tensor pair_mask,
              nv::Tensor mask_argsort,
              int num_activate_out,
              bool is_subm,
              nv::Tensor bias,   // bias 融合进 conv epilogue (空 tensor 表示无 bias)
              bool relu,         // epilogue 是否接 ReLU
              void* stream) {

  int out_channel = filters.shape[0];
  int in_channel = filters.shape[-1];
  int kv = filters.shape[1];

  nv::Tensor out_features;
  if (is_subm) {
    out_features = nv::Tensor::create(std::vector<int64_t>{num_activate_out, out_channel}, features.dtype(), features.device());
  } else {
    out_features = nv::Tensor::create(std::vector<int64_t>{num_activate_out, out_channel}, features.dtype(), features.device());
    out_features.memset(0, stream);
  }

  implicit_gemm_cuda(features, filters, indicePairs, pair_mask, mask_argsort,
                     out_features, bias, relu, stream);

 return out_features;
}

} // namespace spconv
