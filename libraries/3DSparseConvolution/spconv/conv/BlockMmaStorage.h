#pragma once
#include <cuda_fp16.h>
#include "ConvCommon.h"
namespace cumm {
namespace conv {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  aligned_array<half, 4096, 16> smem_A;
  aligned_array<half, 8192, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace conv
} // namespace cumm