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

template <typename layout>
static __device__ __forceinline__ float dot_f8_e4m3(int n, const void * row_data, const float * y) {
    const auto * x = layout::codes(row_data);
    float sum = 0.0f;
    if (n % 2 == 0 && uintptr_t(x) % alignof(uint16_t) == 0 && uintptr_t(y) % alignof(float2) == 0) {
        const auto * y2 = (const float2 *)y;
        float2 sum2 = {};
        for (int j = threadIdx.x; j < n/2; j += 256) {
            const float2 w = ggml_cuda_f8_e4m3x2_to_float2(layout::pair(row_data, 2*j));
            const float scale = layout::dot_scale(row_data, 2*j);
            const float2 a = y2[j];
            sum2.x = fmaf(w.x * scale, a.x, sum2.x);
            sum2.y = fmaf(w.y * scale, a.y, sum2.y);
        }
        sum = sum2.x + sum2.y;
    } else {
        for (int j = threadIdx.x; j < n; j += 256) {
            sum = fmaf(ggml_cuda_f8_e4m3_to_float(layout::code(row_data, j)) * layout::dot_scale(row_data, j), y[j], sum);
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
    return sum * layout::scale(row_data);
}

template <typename layout>
static __global__ void mul_mat_vec_f8_e4m3(int n, int ldc, const ggml_fp8_e4m3_t * x,
        const float * y, float * z, size_t nb11) {
    const void * row = (const char *)x + size_t(blockIdx.x)*layout::row_size(n);
    const float sum = dot_f8_e4m3<layout>(n, row, y + size_t(blockIdx.y)*nb11);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + blockIdx.x] = sum;
}

template <int width>
static __global__ void mul_mat_vec_f8_e4m3_wide(int n, int ldc, const void * vx,
        const float * y, float * z, size_t nb11) {
    const int row = blockIdx.x;
    const void * row_data = (const char *)vx + size_t(row)*ggml_cuda_f8_row::row_size(n);
    const auto * x = ggml_cuda_f8_row::codes(row_data);
    y += size_t(blockIdx.y)*nb11;
    const int peel = (-uintptr_t(x)) % width;
    const int end = peel + (n - peel)/width*width;
    float sums[width] = {};
    for (int j = peel + width*threadIdx.x; j < end; j += width*blockDim.x) {
        uint32_t words[width/4];
        if constexpr (width == 16) {
            const uint4 codes = *(const uint4 *)(x + j);
            words[0] = codes.x; words[1] = codes.y; words[2] = codes.z; words[3] = codes.w;
        } else {
            const uint2 codes = *(const uint2 *)(x + j);
            words[0] = codes.x; words[1] = codes.y;
        }
#pragma unroll
        for (int k = 0; k < width/4; ++k) {
            const float2 w0 = ggml_cuda_f8_e4m3x2_to_float2(words[k]);
            const float2 w1 = ggml_cuda_f8_e4m3x2_to_float2(words[k] >> 16);
            const float4 a = *(const float4 *)(y + j + 4*k);
            sums[4*k+0] = fmaf(w0.x, a.x, sums[4*k+0]);
            sums[4*k+1] = fmaf(w0.y, a.y, sums[4*k+1]);
            sums[4*k+2] = fmaf(w1.x, a.z, sums[4*k+2]);
            sums[4*k+3] = fmaf(w1.y, a.w, sums[4*k+3]);
        }
    }
    float sum = 0;
#pragma unroll
    for (int k = 0; k < width; ++k) sum += sums[k];
    if (threadIdx.x < peel) sum += ggml_cuda_f8_e4m3_to_float(x[threadIdx.x]) * y[threadIdx.x];
    const int tail = end + threadIdx.x;
    if (tail < n) sum += ggml_cuda_f8_e4m3_to_float(x[tail]) * y[tail];
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[8];
    const int lane = threadIdx.x % 32;
    if (lane == 0) tmp[threadIdx.x/32] = sum;
    __syncthreads();
    sum = lane < blockDim.x/32 ? tmp[lane] : 0;
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + row] = sum * ggml_cuda_f8_row::scale(row_data);
}

static __device__ __forceinline__ float dot_f8_e4m3_b128(int n, const void * row, const float * y) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float sum = 0;
    for (int b = warp; b < n/128; b += blockDim.x/32) {
        const void * block = (const char *)row + size_t(b)*132;
        const auto * x = ggml_cuda_f8_block::codes(block);
        uint32_t codes;
        if (uintptr_t(x) % alignof(uint32_t) == 0) codes = *(const uint32_t *)(x + 4*lane);
        else memcpy(&codes, x + 4*lane, sizeof(codes));
        const float2 w0 = ggml_cuda_f8_e4m3x2_to_float2(codes);
        const float2 w1 = ggml_cuda_f8_e4m3x2_to_float2(codes >> 16);
        const float * a = y + b*128 + 4*lane;
        float partial = w0.x*a[0];
        partial = fmaf(w0.y, a[1], partial);
        partial = fmaf(w1.x, a[2], partial);
        partial = fmaf(w1.y, a[3], partial);
        partial = warp_reduce_sum(partial);
        if (lane == 0) sum = fmaf(partial, ggml_cuda_f8_row::scale(block), sum);
    }
    __shared__ float tmp[8];
    if (lane == 0) tmp[warp] = sum;
    __syncthreads();
    sum = lane < blockDim.x/32 ? tmp[lane] : 0;
    return warp_reduce_sum(sum);
}

static __global__ void mul_mat_vec_f8_e4m3_b128(int n, int ldc, const void * x,
        const float * y, float * z, size_t nb11) {
    const void * row = (const char *)x + size_t(blockIdx.x)*ggml_cuda_f8_block::row_size(n);
    const float sum = dot_f8_e4m3_b128(n, row, y + size_t(blockIdx.y)*nb11);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + blockIdx.x] = sum;
}

template <bool blocked>
static __global__ void mul_mat_vec_f8_e4m3_id(int n, int experts, int channels,
        const char * x, const char * y, const char * ids, char * z,
        size_t nb01, size_t nb02, size_t nb11, size_t nb12, size_t nbi0, size_t nbi1, size_t nb1, size_t nb2) {
    const int slot = blockIdx.y;
    const int token = blockIdx.z;
    const int expert = *(const int *)(ids + size_t(token)*nbi1 + size_t(slot)*nbi0);
    if (expert < 0 || expert >= experts) return;
    const void * row = x + size_t(expert)*nb02 + size_t(blockIdx.x)*nb01;
    const auto * a = (const float *)(y + size_t(slot%channels)*nb11 + size_t(token)*nb12);
    float sum;
    if constexpr (blocked) sum = dot_f8_e4m3_b128(n, row, a);
    else sum = dot_f8_e4m3<ggml_cuda_f8_row>(n, row, a);
    if (threadIdx.x == 0) *(float *)(z + size_t(token)*nb2 + size_t(slot)*nb1 + blockIdx.x*sizeof(float)) = sum;
}

template <bool blocked>
static __global__ void mul_mat_vec_f8_e4m3_up_gate_id(int n, int rows, int experts, int channels,
        const char * up, const char * gate, const char * y, const char * ids,
        const char * bias_up, const char * bias_gate, float * dst, ggml_unary_op unary, float limit,
        size_t nbu1, size_t nbu2, size_t nbg1, size_t nbg2, size_t nb11, size_t nb12,
        size_t nbi0, size_t nbi1, size_t nbbu1, size_t nbbg1) {
    const int slot = blockIdx.y;
    const int token = blockIdx.z;
    const int row = blockIdx.x;
    const int expert = *(const int *)(ids + size_t(token)*nbi1 + size_t(slot)*nbi0);
    const size_t out = (size_t(token)*gridDim.y + slot)*rows + row;
    if (expert < 0 || expert >= experts) {
        if (threadIdx.x == 0) dst[out] = 0;
        return;
    }
    const auto * a = (const float *)(y + size_t(slot%channels)*nb11 + size_t(token)*nb12);
    const void * ur = up + size_t(expert)*nbu2 + size_t(row)*nbu1;
    const void * gr = gate + size_t(expert)*nbg2 + size_t(row)*nbg1;
    float su, sg;
    if constexpr (blocked) su = dot_f8_e4m3_b128(n, ur, a);
    else su = dot_f8_e4m3<ggml_cuda_f8_row>(n, ur, a);
    __syncthreads();
    if constexpr (blocked) sg = dot_f8_e4m3_b128(n, gr, a);
    else sg = dot_f8_e4m3<ggml_cuda_f8_row>(n, gr, a);
    if (threadIdx.x == 0) {
        if (bias_up) su += ((const float *)(bias_up + size_t(expert)*nbbu1))[row];
        if (bias_gate) sg += ((const float *)(bias_gate + size_t(expert)*nbbg1))[row];
        switch (unary) {
            case GGML_UNARY_OP_SILU:
                if (limit < 1e-6f) dst[out] = sg * su / (1.0f + expf(-sg));
                else dst[out] = min(sg / (1.0f + expf(-sg)), limit) * max(-limit, min(limit, su));
                break;
            case GGML_UNARY_OP_RELU:
                dst[out] = fmaxf(sg, 0) * su;
                break;
            case GGML_UNARY_OP_GELU:
                dst[out] = 0.5f*sg*su*(1.0f + tanhf(0.79788456080286535587989211986876f*sg*(1.0f + 0.044715f*sg*sg)));
                break;
            case GGML_UNARY_OP_SIGMOID:
                dst[out] = su / (1.0f + expf(-sg));
                break;
            case GGML_UNARY_OP_SWIGLU_OAI:
                sg = fminf(sg, 7.0f);
                su = fmaxf(fminf(su, 7.0f), -7.0f);
                dst[out] = (sg / (1.0f + expf(-sg * 1.702f))) * (1.0f + su);
                break;
            default: assert(false);
        }
    }
}
