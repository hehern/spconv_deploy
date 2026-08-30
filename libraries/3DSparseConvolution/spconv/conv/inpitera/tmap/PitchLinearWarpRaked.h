#pragma once
#include <array>
namespace cumm {
namespace conv {
namespace inpitera {
namespace tmap {
struct PitchLinearWarpRaked {
  __forceinline__ __host__ __device__ static bool is_skipped(int thread_id)   {
    
    if(!0)
        return false;
    return thread_id >= 128;
  }
  __forceinline__ __host__ __device__ static std::array<int, 2> initial_offset(int thread_id)   {
    if(is_skipped(thread_id))
        return {0, 0};        // to InputIter: inefficient but convenience dummy offset.
    int warp_id = (thread_id / 32);
    int lane_id = (thread_id % 32);
    std::array<int, 2> warp_offset{warp_id / 1,
                                warp_id % 1};
    constexpr std::array<int, 2> kWarpDilation{16, 4};
    std::array<int, 2> thread_offset_in_warp{lane_id / 4,
                                            lane_id % 4};
    std::array<int, 2> offset_in_tile;
    offset_in_tile[0] = kWarpDilation[0] * warp_offset[0] + thread_offset_in_warp[0];
    offset_in_tile[1] = kWarpDilation[1] * warp_offset[1] + thread_offset_in_warp[1];
    return {offset_in_tile[0], offset_in_tile[1] * 8};
  }
};
} // namespace inpitera
} // namespace conv
} // namespace cumm