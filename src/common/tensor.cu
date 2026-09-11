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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include "check.hpp"
#include "launch.cuh"
#include "tensor.hpp"

namespace nv {

using namespace std;

// ===== tensor 内存池 =====
// 推理循环中 tensor 创建/销毁极频繁 (Lidar Backbone 每帧约 150 次:
// 21 层 conv 输出/bias 输出 + 8 次 rulebook 生成 + thrust 内部临时分配),
// 裸 cudaMalloc/cudaFree 不仅单次开销大, cudaFree 还会隐式同步设备,
// 打断 kernel 流水线产生 GPU 空闲气泡 (实测 kernel 仅 6ms 时 wall 达 17ms)。
//
// 池策略:
//  - best-fit: 请求取 >= bytes 的最小可用空闲块 (std::map 有序桶 + lower_bound)。
//    好处: prime 预填的通用档位块可以服务任意更小的请求, 第一帧即零 cudaMalloc;
//    点云数量波动导致 size 漂移时也优先复用已有块而非新分配。
//  - 归还按请求 bytes 记账 (块实际 >= bytes, 复用永远安全), 稳态下
//    每帧分配 pattern 重复, 精确桶自然形成, 命中率接近 100%。
//  - pool_prime(): 启动时按通用阶梯一次性预填 (约 260MB),
//    消除第一帧的池冷启动 cudaMalloc; 池内显存由 context 销毁时回收,
//    同时也消除了进程退出时 cudaFree 触发 "driver shutting down" 报错。
namespace {
struct MemoryPool {
  std::mutex mutex;
  std::map<size_t, std::vector<void*>> device_blocks;
  std::map<size_t, std::vector<void*>> host_blocks;
  static constexpr size_t kMaxCachedPerBucket = 64;     // 每桶最多缓存块数
  static constexpr size_t kMaxPooledBytes = 1ull << 28; // 超过 256MB 直接走 CUDA API

  void* acquire(size_t bytes, bool device) {
    if (bytes == 0 || bytes > kMaxPooledBytes) return nullptr;
    auto& pool = device ? device_blocks : host_blocks;
    std::lock_guard<std::mutex> lock(mutex);
    // best-fit: 从小到大找第一个 >= bytes 且非空的桶
    for (auto it = pool.lower_bound(bytes); it != pool.end(); ++it) {
      if (!it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        return p;
      }
    }
    return nullptr;
  }

  void release(void* p, size_t bytes, bool device) {
    if (p == nullptr || bytes == 0 || bytes > kMaxPooledBytes) {
      // 不入池的块保持原语义直接释放
      if (p != nullptr) {
        if (device) checkRuntime(cudaFree(p));
        else checkRuntime(cudaFreeHost(p));
      }
      return;
    }
    auto& pool = device ? device_blocks : host_blocks;
    std::lock_guard<std::mutex> lock(mutex);
    auto& bucket = pool[bytes];
    if (bucket.size() < kMaxCachedPerBucket) {
      bucket.push_back(p);
    } else {
      if (device) checkRuntime(cudaFree(p));
      else checkRuntime(cudaFreeHost(p));
    }
  }

  // 启动时通用阶梯预填: 覆盖 SCN 热路径的尺寸量级
  // (conv 输出 numAct*K*2B ≈ 1~12MB, rulebook 27*numAct*4B ≈ 5~8MB,
  //  hash/argsort/mask ≈ 0.2~0.3MB, bias/relu 输出同量级),
  // best-fit 下这些请求全部命中预填块, 第一帧不再有池冷启动分配。
  // 2026-09-10 实测 (SPCONV_POOL_DUMP): 8MB 桶原 8 块被 4~6MB 规则簿 tensor
  // (帧内同时存活 8+) + thrust 排序临时缓冲耗尽 → miss; dense/transpose 输出
  // 16.59MB > 16MB 块, best-fit 无法命中 → 必 miss。故 8MB 加到 24 块并新增 24MB 档。
  // (同会话对比: 补档后 cudaMalloc 7→0, 空闲 -0.14ms, 窗口 -0.13ms; 早期观察到的
  //  索引 kernel +120µs 差异经 state8 回退实验确认是机器状态漂移, 与补档无关。)
  void prime_device() {
    const std::pair<size_t, int> plan[] = {
        {64ull << 10, 16},    //  64KB ×16 =   1MB
        {256ull << 10, 16},   // 256KB ×16 =   4MB
        {1ull << 20, 16},     //   1MB ×16 =  16MB
        {4ull << 20, 12},     //   4MB ×12 =  48MB
        {8ull << 20, 24},     //   8MB ×24 = 192MB
        {16ull << 20, 8},     //  16MB ×8  = 128MB
        {24ull << 20, 4},     //  24MB ×4  =  96MB  (覆盖 16.59MB dense/transpose 输出)
    };                        // 合计约 485MB
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& item : plan) {
      auto& bucket = device_blocks[item.first];
      for (int i = 0; i < item.second; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, item.first) == cudaSuccess) bucket.push_back(p);
      }
    }
  }
};
MemoryPool& memory_pool() {
  static MemoryPool inst;
  return inst;
}
}  // namespace

// 第三方库 (thrust/cub) 临时缓冲接入内存池的公共入口, 语义与 TensorData::create/free 一致。
void* pool_acquire_device(size_t bytes) { return memory_pool().acquire(bytes, true); }
void pool_release_device(void* ptr, size_t bytes) { memory_pool().release(ptr, bytes, true); }

#define DISPATCH_BY_TYPES(dtype, ...)                  \
  [&]() {                                              \
    switch (dtype) {                                   \
      case DataType::Float32: {                        \
        using scalar_t = float;                        \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Float16: {                        \
        using scalar_t = half;                         \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt32: {                         \
        using scalar_t = uint32_t;                     \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt64: {                         \
        using scalar_t = uint64_t;                     \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt8: {                          \
        using scalar_t = uint8_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int32: {                          \
        using scalar_t = int32_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int64: {                          \
        using scalar_t = int64_t;                      \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int8: {                           \
        using scalar_t = int8_t;                       \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::Int16: {                          \
        using scalar_t = short;                        \
        return __VA_ARGS__();                          \
      }                                                \
      case DataType::UInt16: {                         \
        using scalar_t = unsigned short;               \
        return __VA_ARGS__();                          \
      }                                                \
      default: {                                       \
        using scalar_t = float;                        \
        Assertf(false, "Unknow dtype %d", (int)dtype); \
        return __VA_ARGS__();                          \
      }                                                \
    }                                                  \
  }();

static inline float _native_half2float(const unsigned short h) {
  unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
  unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
  unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
  float f(0.f);
  if (exponent == 0x1fU) { /* NaN or Inf */
    /* discard sign of a NaN */
    sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
    mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
    exponent = 0xffU;
  } else if (exponent == 0U) { /* Denorm or Zero */
    if (mantissa != 0U) {
      unsigned int msb;
      exponent = 0x71U;
      do {
        msb = (mantissa & 0x400000U);
        mantissa <<= 1U; /* normalize */
        --exponent;
      } while (msb == 0U);
      mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70U;
  }
  unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
  memcpy(&f, &u, sizeof(f));
  return f;
}

template <typename _T>
static __global__ void arange_kernel_device(size_t num, _T* pdata) {
  int index = cuda_linear_index;
  if (index < num) {
    pdata[index] = index;
  }
}

template <typename _T>
static void arange_kernel_host(size_t num, _T* pdata) {
  for (size_t index = 0; index < num; ++index) pdata[index] = index;
}

template <>
void arange_kernel_host<half>(size_t num, half* pdata) {
  for (size_t index = 0; index < num; ++index) pdata[index] = half((int)index);
}

template <typename _AData, typename _BData>
static __global__ void any_to_any_device(size_t num, _AData* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = input[index];
  }
}

template <typename _BData>
static __global__ void any_to_any_device(size_t num, int64_t* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = (int)input[index];
  }
}

template <typename _BData>
static __global__ void any_to_any_device(size_t num, uint64_t* input, _BData* output) {
  int index = cuda_linear_index;
  if (index < num) {
    output[index] = (int)input[index];
  }
}

template <>
std::string format_shape(const std::vector<int64_t>& shape) {
  char buf[200] = {0};
  char* p = buf;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + 1 < shape.size())
      p += sprintf(p, "%ld x ", shape[i]);
    else
      p += sprintf(p, "%ld", shape[i]);
  }
  return buf;
}

template <>
std::string format_shape(const std::vector<int>& shape) {
  char buf[200] = {0};
  char* p = buf;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i + 1 < shape.size())
      p += sprintf(p, "%d x ", shape[i]);
    else
      p += sprintf(p, "%d", shape[i]);
  }
  return buf;
}

template <typename T>
std::vector<int64_t> to_int64(const std::vector<T>& array) {
  std::vector<int64_t> output(array.size());
  for (size_t i = 0; i < array.size(); ++i) output[i] = static_cast<int64_t>(array[i]);

  return output;
}

const char* dtype_string(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
      return "Float32";
    case DataType::Float16:
      return "Float16";
    case DataType::Int32:
      return "Int32";
    case DataType::Int64:
      return "Int64";
    case DataType::UInt64:
      return "UInt64";
    case DataType::UInt32:
      return "UInt32";
    case DataType::Int8:
      return "Int8";
    case DataType::UInt8:
      return "UInt8";
    case DataType::Int16:
      return "Int16";
    case DataType::UInt16:
      return "UInt16";
    default:
      return "Unknow";
  }
}

size_t dtype_bytes(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
      return sizeof(float);
    case DataType::Float16:
      return sizeof(unsigned short);
    case DataType::Int32:
      return sizeof(int);
    case DataType::Int64:
      return sizeof(int64_t);
    case DataType::UInt64:
      return sizeof(uint64_t);
    case DataType::UInt32:
      return sizeof(unsigned int);
    case DataType::Int8:
      return sizeof(char);
    case DataType::UInt8:
      return sizeof(unsigned char);
    case DataType::Int16:
      return sizeof(short);
    case DataType::UInt16:
      return sizeof(unsigned short);
    default:
      return 0ul;
  }
}

TensorData::~TensorData() { TensorData::free(); }

void TensorData::free() {
  if (data && owner) {
    // 归还内存池复用 (见文件头部 MemoryPool 说明), 消除热路径 cudaFree 的隐式同步
    memory_pool().release(data, bytes, device);
  }
  data = nullptr;
  owner = false;
  bytes = 0;
  dtype = DataType::None;
  device = false;
}

TensorData* TensorData::reference_new(void* data, size_t bytes, DataType dtype, bool device) {
  TensorData* output = new TensorData();
  output->owner = false;
  output->data = data;
  output->dtype = dtype;
  output->bytes = bytes;
  output->device = device;
  return output;
}

void TensorData::reference(void* data, size_t bytes, DataType dtype, bool device) {
  TensorData::free();
  this->owner = false;
  this->data = data;
  this->dtype = dtype;
  this->bytes = bytes;
  this->device = device;
}

TensorData* TensorData::create(size_t bytes, DataType dtype, bool device) {
  TensorData* output = new TensorData();
  output->owner = true;
  output->dtype = dtype;
  output->bytes = bytes;
  output->device = device;

  // 优先从内存池取 (推理时每帧分配 pattern 重复, 命中率接近 100%)
  if (bytes > 0) {
    output->data = memory_pool().acquire(bytes, device);
    if (output->data != nullptr) return output;
  }

  if (device)
    checkRuntime(cudaMalloc(&output->data, bytes));
  else
    checkRuntime(cudaMallocHost(&output->data, bytes));
  return output;
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, bool device) {
  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->shape = shape;
  this->numel = volumn;
  this->ndim = shape.size();
  this->data.reset(TensorData::create(bytes, dtype, device));
}

Tensor::Tensor(std::vector<int32_t> shape, DataType dtype, bool device) : Tensor(to_int64(shape), dtype, device) {}

void Tensor::create_(vector<int64_t> shape, DataType dtype, bool device) {
  this->release();
  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->shape = shape;
  this->numel = volumn;
  this->ndim = shape.size();
  this->data.reset(TensorData::create(bytes, dtype, device));
}

void Tensor::release() {
  this->shape.clear();
  this->data.reset();
  this->numel = 0;
  this->ndim = 0;
}

Tensor Tensor::create(vector<int64_t> shape, DataType dtype, bool device) { return Tensor(shape, dtype, device); }
Tensor Tensor::create(vector<int32_t> shape, DataType dtype, bool device) { return create(to_int64(shape), dtype, device); }

void Tensor::reference(void* data, vector<int64_t> shape, DataType dtype, bool device) {
  if (this->data == nullptr) {
    this->data.reset(new TensorData());
  }

  size_t volumn = std::accumulate(shape.begin(), shape.begin() + shape.size(), 1, std::multiplies<int>());
  size_t bytes = volumn * dtype_bytes(dtype);
  this->data->reference(data, bytes, dtype, device);
  this->numel = volumn;
  this->shape = shape;
  this->ndim = shape.size();
}

void Tensor::reference(void* data, vector<int32_t> shape, DataType dtype, bool device) {
  reference(data, to_int64(shape), dtype, device);
}

Tensor Tensor::from_data(void* data, vector<int64_t> shape, DataType dtype, bool device, void* stream) {
  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  } else {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyHostToHost, (cudaStream_t)stream));
  }
  return output;
}

Tensor Tensor::from_data(void* data, vector<int32_t> shape, DataType dtype, bool device, void* stream) {
  return from_data(data, to_int64(shape), dtype, device, stream);
}

Tensor Tensor::from_data(const void* data, vector<int64_t> shape, DataType dtype, bool device, void* stream) {
  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  } else {
    checkRuntime(cudaMemcpyAsync(output.ptr(), data, output.bytes(), cudaMemcpyHostToHost, (cudaStream_t)stream));
  }
  return output;
}

Tensor Tensor::from_data(const void* data, vector<int32_t> shape, DataType dtype, bool device, void* stream) {
  return from_data(data, to_int64(shape), dtype, device, stream);
}

Tensor Tensor::from_data_reference(void* data, vector<int64_t> shape, DataType dtype, bool device) {
  Tensor output;
  output.reference(data, shape, dtype, device);
  return output;
}

Tensor Tensor::from_data_reference(void* data, vector<int32_t> shape, DataType dtype, bool device) {
  return from_data_reference(data, to_int64(shape), dtype, device);
}

void Tensor::to_device_(void* stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  if (!this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), true));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    this->data = newdata;
  }
}

Tensor Tensor::to_device(void* stream_) const {
  if (!this->device() && !this->empty()) {
    cudaStream_t stream = (cudaStream_t)stream_;
    Tensor output(shape, this->dtype(), true);
    checkRuntime(cudaMemcpyAsync(output.ptr(), this->ptr(), this->bytes(), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    return output;
  }
  return *this;
}

void Tensor::to_host_(void* stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  if (this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    this->data = newdata;
  }
}

Tensor Tensor::to_host(void* stream_) const {
  if (this->device() && !this->empty()) {
    cudaStream_t stream = (cudaStream_t)stream_;
    Tensor output(shape, this->dtype(), false);
    checkRuntime(cudaMemcpyAsync(output.ptr(), this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    return output;
  }
  return *this;
}

void Tensor::arange(void* _stream) {
  if (this->empty()) return;

  cudaStream_t stream = (cudaStream_t)_stream;
  if (this->device()) {
    DISPATCH_BY_TYPES(this->dtype(),
                      [&] { cuda_linear_launch(arange_kernel_device, stream, this->numel, this->ptr<scalar_t>()); });
  } else {
    DISPATCH_BY_TYPES(this->dtype(), [&] { arange_kernel_host(this->numel, this->ptr<scalar_t>()); });
  }
}

void Tensor::memset(unsigned char value, void* stream) {
  if (this->empty()) return;

  if (this->device()) {
    checkRuntime(cudaMemsetAsync(this->ptr(), value, this->bytes(), (cudaStream_t)stream));
  } else {
    ::memset(this->ptr(), value, this->bytes());
  }
}

template <typename T>
void Tensor::fill(const T value, void* stream) {
  if (this->empty()) return;

  if (this->device()) {
    // 值可表示为字节模式 (0x00/0xFF 等) 时走 cudaMemsetAsync: 热路径上所有 fill
    // (indicePairs.fill(-1) / pair_mask.fill(0) / cnt.fill(0) / relu/dense 的 fill(0))
    // 都满足, 用 thrust::fill 时实测每次调用后都会隐式 cudaStreamSynchronize,
    // 打断推理流流水线 (单帧 29 次同步中 25 次由此产生)。同流串行下 memset 与
    // thrust 语义等价, 但只入队一次 cudaMemsetAsync, 无同步、无 thrust 派发开销。
    const unsigned char* v = reinterpret_cast<const unsigned char*>(&value);
    bool byte_pattern = true;
    for (size_t i = 1; i < sizeof(T); ++i) {
      if (v[i] != v[0]) { byte_pattern = false; break; }
    }
    if (byte_pattern) {
      checkRuntime(cudaMemsetAsync(this->ptr(), v[0], this->bytes(), (cudaStream_t)stream));
      return;
    }
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(this->ptr<T>());
    thrust::fill(thrust::cuda::par.on((cudaStream_t)stream), dev_ptr, dev_ptr+this->numel, value);
  } else {
    std::fill(this->ptr<T>(), this->ptr<T>()+this->numel, value);
  }
}
template void Tensor::fill<int32_t>(const int32_t, void*);
template void Tensor::fill<uint32_t>(const uint32_t, void*);
template void Tensor::fill<half>(const half, void*);

Tensor Tensor::loadbinary(const std::string& file, std::vector<int64_t> shape, DataType dtype, bool device) {
  FILE* f = fopen(file.c_str(), "rb");
  if (f == nullptr) return Tensor();

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  int num_implicit_dim = 0;
  size_t volumn_explicit_dim = 1;
  int i_implicit_dim = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      num_implicit_dim++;
      i_implicit_dim = i;
    } else
      volumn_explicit_dim *= shape[i];
  }

  if (num_implicit_dim > 0) {
    if (num_implicit_dim != 1) {
      printf("%d implicit dimensions are found, Only one implicit dimension can be supported.\n",
             num_implicit_dim);
      fclose(f);
      return Tensor();
    }

    size_t explicit_bytes = dtype_bytes(dtype) * volumn_explicit_dim;
    if (fsize % explicit_bytes != 0) {
      printf(
          "Cannot be calculated the implicit dimension, and the file bytes[%ld] cannot be divided "
          "by the size of the "
          "explicit dimension bytes[%ld]\n",
          fsize, explicit_bytes);
      fclose(f);
      return Tensor();
    }
    shape[i_implicit_dim] = fsize / explicit_bytes;
    volumn_explicit_dim *= shape[i_implicit_dim];
  }

  size_t bytes = dtype_bytes(dtype) * volumn_explicit_dim;
  if (bytes != fsize) {
    printf(
        "Cannot be loaded by the specified shape, The file has %ld byte and the shape requires %ld "
        "byte\n",
        fsize, bytes);
    fclose(f);
    return Tensor();
  }

  vector<unsigned char> host_data(bytes);
  if (fread(host_data.data(), 1, bytes, f) != bytes) {
    printf("Failed to read %ld bytes in file: %s\n", bytes, file.c_str());
    fclose(f);
    return Tensor();
  }
  fclose(f);

  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToDevice));
  } else {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToHost));
  }
  checkRuntime(cudaDeviceSynchronize());
  return output;
}

Tensor Tensor::load(const std::string& file, bool device) {
  FILE* f = fopen(file.c_str(), "rb");
  if (f == nullptr) return Tensor();

  int head[3];
  if (fread(head, 1, sizeof(head), f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  if (head[0] != 0x33ff1101) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  int ndim = head[1];
  int dtypei = head[2];
  int dims[16];

  if (fread(dims, 1, ndim * sizeof(int), f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  vector<int64_t> shape(ndim);
  std::transform(dims, dims + ndim, shape.begin(), [](int x) { return x; });

  int volumn = std::accumulate(dims, dims + ndim, 1, std::multiplies<int>());
  DataType dtype = (DataType)dtypei;
  size_t bytes = dtype_bytes(dtype) * volumn;
  vector<unsigned char> host_data(bytes);

  if (fread(host_data.data(), 1, bytes, f) == 0) {
    printf("This is invalid tensor file %s\n", file.c_str());
    fclose(f);
    return Tensor();
  }

  fclose(f);

  Tensor output = Tensor::create(shape, dtype, device);
  if (device) {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToDevice));
  } else {
    checkRuntime(cudaMemcpy(output.ptr(), host_data.data(), bytes, cudaMemcpyHostToHost));
  }
  checkRuntime(cudaDeviceSynchronize());
  return output;
}

void Tensor::print(const char* prefix, size_t offset, size_t num_per_line, size_t lines) const {
  printf("%s[%s] %s%s", prefix, dtype_string(dtype()), format_shape(shape).c_str(), lines == 1 ? ": " : ": \n");

  if (this->empty()) {
    printf("empty.\n");
    return;
  }

  shared_ptr<TensorData> tensor_data = this->data;
  if (this->device()) {
    tensor_data = shared_ptr<TensorData>(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpy(tensor_data->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToHost));
  }

  size_t num_print = min(lines * num_per_line, numel);
  if (this->dtype() == DataType::Float32) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%.3f ", *((float*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Float16) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%.3f ", _native_half2float(*((unsigned short*)tensor_data->data + offset + i)));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int32 || this->dtype() == DataType::UInt32) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%d ", *((int*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int64 || this->dtype() == DataType::UInt64) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%ld ", *((int64_t*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  } else if (this->dtype() == DataType::Int16 || this->dtype() == DataType::UInt16) {
    for (size_t i = 0; i < num_print; ++i) {
      printf("%d ", *((short*)tensor_data->data + offset + i));
      if ((i + 1) % num_per_line == 0) printf("\n");
    }
  }
  if (num_print % num_per_line != 0) printf("\n");
}

void Tensor::copy_from_host(const void* data, void* stream) {
  if (this->empty()) return;

  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  if (this->device()) {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyHostToDevice, _stream));
  } else {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyHostToHost, _stream));
  }
}

void Tensor::copy_from_device(const void* data, void* stream) {
  if (this->empty()) return;

  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  if (this->device()) {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyDeviceToDevice, _stream));
  } else {
    checkRuntime(cudaMemcpyAsync(this->ptr(), data, this->bytes(), cudaMemcpyDeviceToHost, _stream));
  }
}

Tensor Tensor::clone(void* _stream) const {
  Tensor output = *this;
  cudaStream_t stream = (cudaStream_t)_stream;
  if (this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), true));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyDeviceToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    output.data = newdata;
  } else if (!this->device() && !this->empty()) {
    shared_ptr<TensorData> newdata(TensorData::create(this->bytes(), this->dtype(), false));
    checkRuntime(cudaMemcpyAsync(newdata->data, this->ptr(), this->bytes(), cudaMemcpyHostToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    output.data = newdata;
  }
  return output;
}

Tensor Tensor::to_half(void* _stream) const {
  Tensor output;
  cudaStream_t stream = (cudaStream_t)_stream;

  if (this->empty()) return output;
  if (this->dtype() == DataType::Float16) return this->clone(stream);
  if (!this->device()) {
    printf("Unsupport non device convertion.\n");
    return output;
  }
  output = Tensor::create(this->shape, DataType::Float16);

  DISPATCH_BY_TYPES(this->dtype(), [&] {
    cuda_linear_launch(any_to_any_device, stream, this->numel, this->ptr<scalar_t>(), output.ptr<half>());
    checkRuntime(cudaStreamSynchronize(stream));
  });
  return output;
}

Tensor Tensor::to_float(void* _stream) const {
  Tensor output;
  cudaStream_t stream = (cudaStream_t)_stream;

  if (this->empty()) return output;
  if (this->dtype() == DataType::Float32) return this->clone(stream);
  if (!this->device()) {
    printf("Unsupport non device convertion.\n");
    return output;
  }
  output = Tensor::create(this->shape, DataType::Float32);

  DISPATCH_BY_TYPES(this->dtype(), [&] {
    cuda_linear_launch(any_to_any_device, stream, this->numel, this->ptr<scalar_t>(), output.ptr<float>());
    checkRuntime(cudaStreamSynchronize(stream));
  });
  return output;
}

bool Tensor::save(const std::string& file, void* stream_) const {
  cudaStream_t stream = (cudaStream_t)stream_;
  FILE* f = fopen(file.c_str(), "wb");
  if (f == nullptr) {
    printf("Failed to open %s\n", file.c_str());
    return false;
  }

  int head[] = {0x33ff1101, (int)this->shape.size(), (int)this->dtype()};
  int dims[16];
  std::transform(this->shape.begin(), this->shape.end(), dims, [](int64_t i) -> int { return i; });

  fwrite(head, 1, sizeof(head), f);
  fwrite(dims, 1, this->shape.size() * sizeof(int), f);

  if (this->device()) {
    std::vector<char> host_data(this->bytes());
    checkRuntime(cudaMemcpyAsync(host_data.data(), this->ptr(), this->bytes(), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    fwrite(host_data.data(), 1, this->bytes(), f);
  } else {
    fwrite(this->ptr(), 1, this->bytes(), f);
  }
  fclose(f);
  return true;
}

void Tensor::self_byte_check(size_t type_bytes) const {
  size_t self_bytes = dtype_bytes(this->dtype());
  if (self_bytes != type_bytes) {
    this->print("This");
    Assertf(self_bytes == type_bytes,
            "Failed to check the data type, your code may have a logic error. The type of this tensor is %d bytes, but the "
            "pointer is %d bytes.",
            static_cast<int>(self_bytes), static_cast<int>(type_bytes));
  }
}

bool Tensor::save(const Tensor& tensor, const std::string& file, void* stream) { return tensor.save(file, stream); }

void Tensor::pool_prime() { memory_pool().prime_device(); }

};  // namespace nv