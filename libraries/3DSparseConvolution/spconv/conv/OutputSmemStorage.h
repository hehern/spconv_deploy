#pragma once
#include <cuda_fp16.h>
#include "ConvCommon.h"
namespace cumm {
namespace conv {
namespace out_smem_storage {
struct OutputSmemStorage {
  aligned_array<half, 2176, 16> smem;
};
} // namespace out_smem_storage
} // namespace conv
} // namespace cumm