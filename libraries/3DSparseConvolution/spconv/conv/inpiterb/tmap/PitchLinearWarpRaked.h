#pragma once
#include <array>
namespace cumm {
namespace conv {
namespace inpiterb {
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
    // tnt(kForward): warp dilation {32, 4}, lane 映射 lane_id/4, lane_id%4
    constexpr std::array<int, 2> kWarpDilation{32, 4};//B-tile数据为128×32,每个warp处理分4次迭代处理32行数据，即每次迭代处理8行，每个lane处理8个数据
    std::array<int, 2> thread_offset_in_warp{lane_id / 4,//一个warp处理8行
                                            lane_id % 4};//一行数据4个lane
    // offset_in_tile = kWarpDilation * warp_offset + thread_offset_in_warp
    std::array<int, 2> offset_in_tile;
    offset_in_tile[0] = kWarpDilation[0] * warp_offset[0] + thread_offset_in_warp[0];
    offset_in_tile[1] = kWarpDilation[1] * warp_offset[1] + thread_offset_in_warp[1];
    return {offset_in_tile[0], offset_in_tile[1] * 8};
  }
};
} // namespace tmap
} // namespace inpiterb
} // namespace conv
} // namespace cumm
