#pragma once

#include "common.cuh"

static __device__ __forceinline__ float ggml_cuda_f8_e4m3_to_float(ggml_fp8_e4m3_t x) {
    const uint32_t sign = (uint32_t)(x.bits & 0x80) << 24;
    const uint32_t mag = x.bits & 0x7f;
    if (mag == 0x7f) {
        return __uint_as_float(sign | 0x7fc00000);
    }
    if (mag < 8) {
        return __uint_as_float(sign | __float_as_uint(mag * 0.001953125f));
    }
    return __uint_as_float(sign | ((mag / 8 + 120) << 23) | ((mag & 7) << 20));
}
