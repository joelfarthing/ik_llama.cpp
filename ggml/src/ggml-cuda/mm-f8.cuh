#pragma once

#include "mmv-f8.cuh"
#include "mma_new.cuh"

template <typename layout>
static __global__ void mul_mat_f8_e4m3(int n, int rows, int cols, int ldc,
        const char * x, const half * y, float * z) {
#ifdef INT8_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr bool blocked = std::is_same_v<layout, ggml_cuda_f8_block>;
    constexpr int stride = 68;
    __shared__ half2 a[64*stride];
    __shared__ half2 b[64*stride];
    __shared__ float scales[64];
    const int tid = 32*threadIdx.y + threadIdx.x;
    const int first_row = 64*blockIdx.x;
    const int first_col = 64*blockIdx.y;
    const int warp_row = 16*(threadIdx.y/2);
    const int warp_col = 32*(threadIdx.y%2);
    tile<16, 8, float> total[4];
    for (int k0 = 0; k0 < n; k0 += 128) {
        for (int i = tid; i < 64*64; i += 256) {
            const int r = i/64;
            const int k = 2*(i%64);
            half2 value = make_half2(0.0f, 0.0f);
            if (first_row + r < rows) {
                const void * row = x + size_t(first_row + r)*layout::row_size(n);
                const uint16_t codes = uint16_t(layout::code(row, k0+k).bits) |
                                       (uint16_t(layout::code(row, k0+k+1).bits) << 8);
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
                value = __nv_cvt_fp8x2_to_halfraw2(codes, __NV_E4M3);
#else
                const float2 v = ggml_cuda_f8_e4m3x2_to_float2(codes);
                value = make_half2(v.x, v.y);
#endif
            }
            a[r*stride + k/2] = value;
            b[r*stride + k/2] = first_col + r < cols ?
                *(const half2 *)(y + size_t(first_col+r)*n + k0+k) : make_half2(0.0f, 0.0f);
        }
        if (tid < 64) {
            scales[tid] = first_row + tid < rows ? layout::scale(x + size_t(first_row+tid)*layout::row_size(n), k0) : 0;
        }
        __syncthreads();
        tile<16, 8, float> partial[4];
#pragma unroll
        for (int k = 0; k < 64; k += 8) {
            tile<16, 8, half2> at;
            load_ldmatrix(at, a + warp_row*stride + k, stride);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                tile<8, 8, half2> bt;
                load_ldmatrix(bt, b + (warp_col+8*j)*stride + k, stride);
                if constexpr (blocked) mma(partial[j], at, bt);
                else mma(total[j], at, bt);
            }
        }
        if constexpr (blocked) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int l = 0; l < total[j].ne; ++l) total[j].x[l] = fmaf(partial[j].x[l], scales[warp_row+total[j].get_i(l)], total[j].x[l]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int l = 0; l < total[j].ne; ++l) {
            const int row = first_row + warp_row + total[j].get_i(l);
            const int col = first_col + warp_col + 8*j + total[j].get_j(l);
            if (row < rows && col < cols) {
                float value = total[j].x[l];
                if constexpr (!blocked) value *= scales[warp_row+total[j].get_i(l)];
                z[size_t(col)*ldc + row] = value;
            }
        }
    }
#else
    GGML_UNUSED(n); GGML_UNUSED(rows); GGML_UNUSED(cols); GGML_UNUSED(ldc);
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(z);
    NO_DEVICE_CODE;
#endif
}
