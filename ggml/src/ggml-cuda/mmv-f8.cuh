#pragma once

#include "f8.cuh"

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

static __device__ __forceinline__ float2 ggml_cuda_f8_e4m3x2_to_float2(uint16_t x) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
    return __half22float2(__nv_cvt_fp8x2_to_halfraw2(x, __NV_E4M3));
#else
    return make_float2(ggml_cuda_f8_e4m3_to_float({uint8_t(x)}),
                      ggml_cuda_f8_e4m3_to_float({uint8_t(x >> 8)}));
#endif
}

static __global__ void mul_mat_vec_f8_e4m3(int n, int ldc, const ggml_fp8_e4m3_t * x,
        const float * y, float * z, size_t nb11, float scale) {
    const int row = blockIdx.x;
    const int col = blockIdx.y;
    x += size_t(row) * n;
    y += size_t(col) * nb11;
    float sum = 0.0f;
    if (n % 2 == 0 && uintptr_t(x) % alignof(uint16_t) == 0 && uintptr_t(y) % alignof(float2) == 0) {
        const auto * x2 = (const uint16_t *)x;
        const auto * y2 = (const float2 *)y;
        float2 sum2 = {};
        for (int j = threadIdx.x; j < n/2; j += 256) {
            const float2 w = ggml_cuda_f8_e4m3x2_to_float2(x2[j]);
            const float2 a = y2[j];
            sum2.x = fmaf(w.x, a.x, sum2.x);
            sum2.y = fmaf(w.y, a.y, sum2.y);
        }
        sum = sum2.x + sum2.y;
    } else {
        for (int j = threadIdx.x; j < n; j += 256) {
            sum = fmaf(ggml_cuda_f8_e4m3_to_float(x[j]), y[j], sum);
        }
    }
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[256/WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) tmp[warp] = sum;
    __syncthreads();
    sum = lane < 256/WARP_SIZE ? tmp[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        z[size_t(col)*ldc + row] = sum * scale;
    }
}
