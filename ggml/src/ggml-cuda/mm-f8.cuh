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

template <int bytes>
static __device__ __forceinline__ void f8_pp_copy(void * dst, const void * src, int count) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
    const uint32_t address = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::
        "r"(address), "l"(src), "n"(bytes), "r"(count));
#else
    if (count) memcpy(dst, src, count);
    if (count < bytes) memset((char *)dst + count, 0, bytes-count);
#endif
}

template <typename layout, int BN, int NW, int stages, bool tail = true, int BK = 64>
static __global__ void __launch_bounds__(NW*32, 1) mul_mat_f8_pp(
        int n, int rows, int cols, int ldc, const char * __restrict__ x, const half * __restrict__ y, float * __restrict__ z) {
#ifdef INT8_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr bool blocked = std::is_same_v<layout, ggml_cuda_f8_block>;
    constexpr int BM = 128, stride = BK/2+4, code_stride = BK/4+4, WC = NW/4, NJ = BN/WC/8;
    extern __shared__ __align__(16) uint32_t storage[];
    uint32_t * codes = storage;
    half2 * b = (half2 *)(codes + stages*BM*code_stride);
    float * scales = (float *)(b + stages*BN*stride);
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int tile_index = blockIdx.x + blockIdx.y*gridDim.x;
    const int group = tile_index/(8*gridDim.y);
    const int group_rows = min(8, int(gridDim.x)-8*group);
    const int first_row = (8*group + (tile_index%(8*gridDim.y))%group_rows)*BM;
    const int first_col = ((tile_index%(8*gridDim.y))/group_rows)*BN;
    const int warp_row = (threadIdx.y/WC)*32, warp_col = (threadIdx.y%WC)*(BN/WC);
    const size_t row_bytes = layout::row_size(n);
    auto load = [&](int k0, int s) {
        if (k0 < n) {
            if constexpr (!tail) {
                for (int i = tid; i < BM*(BK/16+1); i += NW*32) {
                    const int r = i/(BK/16+1), j = i%(BK/16+1);
                    const char * row = x + size_t(first_row+r)*row_bytes;
                    const char * begin;
                    if constexpr (blocked) begin = row + size_t(k0/128)*132 + 4 + k0%128;
                    else begin = row + 4 + k0;
                    const char * src = (const char *)(uintptr_t(begin) & ~uintptr_t(15)) + 16*j;
                    f8_pp_copy<16>(codes + (s*BM+r)*code_stride + 4*j, src, min(16, int(begin+BK-src)));
                }
            } else {
                for (int i = tid; i < BM*BK/4; i += NW*32) {
                    const int r = i/(BK/4), k = k0 + 4*(i%(BK/4));
                    uint32_t * dst = codes + (s*BM+r)*code_stride + i%(BK/4);
                    const bool valid = !tail || (first_row + r < rows && k + 3 < n);
                    const char * row = x + size_t(first_row + r)*row_bytes;
                    const char * src = x;
                    if (valid) {
                        if constexpr (blocked) src = row + size_t(k/128)*132 + 4 + k%128;
                        else src = row + 4 + k;
                    }
                    if (tail && valid && uintptr_t(src)%4 != 0) {
                        uint32_t v;
                        memcpy(&v, src, 4);
                        *dst = v;
                    } else if (tail && first_row + r < rows && k < n && !valid) {
                        uint32_t v = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) if (k+j < n) v |= uint32_t(layout::code(row, k+j).bits) << (8*j);
                        *dst = v;
                    } else {
                        f8_pp_copy<4>(dst, src, valid ? 4 : 0);
                    }
                }
            }
            for (int i = tid; i < BN*BK/8; i += NW*32) {
                const int c = i/(BK/8), k = k0 + 8*(i%(BK/8));
                half * dst = (half *)(b + (s*BN+c)*stride + (k-k0)/2);
                const bool valid = !tail || (first_col+c < cols && k+7 < n);
                const half * src = valid ? y + size_t(first_col+c)*n + k : y;
                if (valid && (!tail || uintptr_t(src)%16 == 0)) {
                    f8_pp_copy<16>(dst, src, 16);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) dst[j] = first_col+c < cols && k+j < n ? y[size_t(first_col+c)*n+k+j] : __float2half(0);
                }
            }
            if constexpr (blocked) {
                for (int r = tid; r < BM; r += NW*32) {
                    scales[s*BM+r] = (!tail || first_row+r < rows) ? layout::scale(x + size_t(first_row+r)*row_bytes, k0) : 0;
                }
            }
        }
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
#endif
    };
#pragma unroll
    for (int s = 0; s < stages-1; ++s) load(s*BK, s);
    tile<16, 8, float> total[2][NJ], partial[2][NJ];
    for (int k0 = 0, step = 0; k0 < n; k0 += BK, ++step) {
        const int s = step%stages;
        load(k0 + (stages-1)*BK, (step+stages-1)%stages);
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group %0;" :: "n"(stages-1));
#endif
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK/2; k += 8) {
            tile<16, 8, half2> at[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int l = 0; l < at[i].ne; ++l) {
                    const int row = warp_row+16*i+at[i].get_i(l);
                    int shift = 0;
                    if constexpr (!tail) {
                        const char * ptr = x + size_t(first_row+row)*row_bytes;
                        if constexpr (blocked) ptr += size_t(k0/128)*132 + 4 + k0%128;
                        else ptr += 4 + k0;
                        shift = (uintptr_t(ptr)%16)/2;
                    }
                    const uint16_t v = ((const uint16_t *)(codes + (s*BM+row)*code_stride))[shift+k+at[i].get_j(l)];
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
                    at[i].x[l] = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
#else
                    const float2 f = ggml_cuda_f8_e4m3x2_to_float2(v);
                    at[i].x[l] = make_half2(f.x, f.y);
#endif
                }
            }
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                tile<8, 8, half2> bt;
                load_ldmatrix(bt, b + (s*BN+warp_col+8*j)*stride+k, stride);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if constexpr (blocked) mma(partial[i][j], at[i], bt);
                    else mma(total[i][j], at[i], bt);
                }
            }
        }
        if constexpr (blocked) {
            if (step%(128/BK) == 128/BK-1 || k0+BK >= n) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
#pragma unroll
                        for (int l = 0; l < total[i][j].ne; ++l) {
                            total[i][j].x[l] = fmaf(partial[i][j].x[l], scales[s*BM+warp_row+16*i+total[i][j].get_i(l)], total[i][j].x[l]);
                            partial[i][j].x[l] = 0;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int l = 0; l < total[i][j].ne; ++l) {
                const int row = first_row + warp_row + 16*i + total[i][j].get_i(l);
                const int col = first_col + warp_col + 8*j + total[i][j].get_j(l);
                if (row < rows && col < cols) {
                    float value = total[i][j].x[l];
                    if constexpr (!blocked) value *= layout::scale(x + size_t(row)*row_bytes);
                    z[size_t(col)*ldc+row] = value;
                }
            }
        }
    }
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
#else
    GGML_UNUSED(n); GGML_UNUSED(rows); GGML_UNUSED(cols); GGML_UNUSED(ldc);
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(z);
    NO_DEVICE_CODE;
#endif
}

template <int BN, int stages, int BK = 64>
static constexpr int f8_pp_shared() {
    return 4*stages*(128*(BK/4+4)+BN*(BK/2+4)+128);
}

template <typename layout, int BN, int NW, int stages, int BK>
static bool launch_f8_pp(int shared_limit, int n, int rows, int cols, int ldc, const char * x, const half * y, float * z, cudaStream_t stream) {
    constexpr int bytes = f8_pp_shared<BN, stages, BK>();
    if (bytes > shared_limit) return false;
    const bool tail = n%BK != 0 || rows%128 != 0 || cols%BN != 0 || uintptr_t(x)%16 != 0 || uintptr_t(y)%16 != 0;
    auto kernel = tail ? mul_mat_f8_pp<layout, BN, NW, stages, true, BK> : mul_mat_f8_pp<layout, BN, NW, stages, false, BK>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<dim3((rows+127)/128, (cols+BN-1)/BN), dim3(32, NW), bytes, stream>>>(n, rows, cols, ldc, x, y, z);
    return true;
}

template <typename layout>
static bool launch_f8_pp(int cc, int shared_limit, int n, int rows, int cols, int ldc, const char * x, const half * y, float * z, cudaStream_t stream) {
    if (cc >= 1200) {
        if constexpr (std::is_same_v<layout, ggml_cuda_f8_block>) {
            return launch_f8_pp<layout, 128, 8, 1, 128>(shared_limit, n, rows, cols, ldc, x, y, z, stream);
        }
        return launch_f8_pp<layout, 128, 8, 2, 64>(shared_limit, n, rows, cols, ldc, x, y, z, stream);
    } else if (cc == CC_ADA_LOVELACE && std::is_same_v<layout, ggml_cuda_f8_block>) {
        return launch_f8_pp<layout, 128, 8, 1, 128>(shared_limit, n, rows, cols, ldc, x, y, z, stream);
    } else {
        return launch_f8_pp<layout, 64, 4, 2, 64>(shared_limit, n, rows, cols, ldc, x, y, z, stream);
    }
}
