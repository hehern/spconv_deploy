#pragma once
#include <array>
#include "ConvCommon.h"
namespace cumm {
namespace conv {

struct ConvUtils {
  TV_HOST_DEVICE_INLINE static std::array<int, 3> get_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n, int split_k_slices, int kv)   {
    
    std::array<int, 3> grid_dims;

    grid_dims[1] = div_up(n, tile_n);
    grid_dims[0] = div_up(m, tile_m);
    grid_dims[2] = split_k_slices;
    return grid_dims;
  }
};

} // namespace conv
} // namespace cumm