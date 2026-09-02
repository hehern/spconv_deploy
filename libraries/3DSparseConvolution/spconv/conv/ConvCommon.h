#pragma once
#include <cstddef>
#include <cstdint>
#include <array>

namespace cumm {
namespace conv {
#ifdef __CUDACC__
#define TV_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define TV_HOST_DEVICE_INLINE inline
#endif

#define div_up(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

// 固定大小数组, 支持对齐参数 (替代 tv::array / tv::alignedarray)
template <typename T, std::size_t N, std::size_t Align = 0>
struct aligned_array {
    static_assert(Align == 0 || Align >= alignof(T), "invalid alignment");
    static_assert(N > 0, "size must be > 0");
    typedef T value_type;
    alignas(Align) T data[N];
    TV_HOST_DEVICE_INLINE T& operator[](std::size_t i) { return data[i]; }
    TV_HOST_DEVICE_INLINE const T& operator[](std::size_t i) const { return data[i]; }
    TV_HOST_DEVICE_INLINE T* data_() { return data; }
    TV_HOST_DEVICE_INLINE const T* data_() const { return data; }
    TV_HOST_DEVICE_INLINE T* begin() { return data; }
    TV_HOST_DEVICE_INLINE T* end() { return data + N; }
    TV_HOST_DEVICE_INLINE const T* begin() const { return data; }
    TV_HOST_DEVICE_INLINE const T* end() const { return data + N; }
    TV_HOST_DEVICE_INLINE constexpr std::size_t size() const { return N; }
    TV_HOST_DEVICE_INLINE void fill(const T& v) {
        for (std::size_t i = 0; i < N; ++i) data[i] = v;
    }
    TV_HOST_DEVICE_INLINE void clear() { fill(T{}); }
};

// shared memory 指针转换 (替代 tv::gemm::get_smem_pointer)
#if defined(__CUDACC__)
__forceinline__ __device__ unsigned get_smem_pointer(void const* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// 16B 条件全局内存 load (替代 tv::gemm::global_load<alignedarray<int4,1,16>, 16>)
__forceinline__ __device__ void global_load16(void* dst, void const* ptr, bool pred) {
    uint32_t* frag_ptr = reinterpret_cast<uint32_t*>(dst);
    #if (CUDA_VERSION >= 11040 && (__CUDA_ARCH__ >= 750))
      asm volatile (
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.L2::128B.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #else
      asm volatile (
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #endif
}

// 16B 条件全局内存 store (替代 tv::gemm::global_store<alignedarray<int4,1,16>, 16>)
__forceinline__ __device__ void global_store16(void const* src, void* ptr, bool pred) {
    uint32_t const* frag_ptr = reinterpret_cast<uint32_t const*>(src);
    asm volatile (
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p,%4,0;\n"
        "  @p st.global.v4.b32 [%5], {%0, %1, %2, %3};\n"
        "}\n"
        :: "r"(frag_ptr[0]), "r"(frag_ptr[1]), "r"(frag_ptr[2]), "r"(frag_ptr[3]),
           "r"((int)pred), "l"(ptr)
    );
}
#endif

} // namespace conv
} // namespace cumm
