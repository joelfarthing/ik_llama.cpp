#pragma once

#include "mm-f8.cuh"
#include "quantize.cuh"

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
#define GGML_CUDA_F8_LT
#include <cublasLt.h>
#include <map>

static bool f8_lt_eligible(const ggml_tensor * node, int cc) {
    if (node->op != GGML_OP_MUL_MAT || node->op_params[0] != GGML_PREC_DEFAULT) return false;
    const auto * w = node->src[0];
    const auto * a = node->src[1];
    return w->type == GGML_TYPE_F8_E4M3_KS && f8_w8a8(w, cc) &&
        a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && a->ne[1] > 8 &&
        w->ne[0]%16 == 0 && w->ne[1]%4 == 0 &&
        w->ne[2]*w->ne[3] == 1 && a->ne[2]*a->ne[3] == 1 &&
        ggml_is_contiguous(w) && ggml_is_contiguous(node) &&
        uintptr_t(w->data)%16 == 0 && uintptr_t(node->data)%16 == 0;
}

static __global__ void f8_lt_quantize(const float * input, uint8_t * output, float * scales, int k, size_t stride) {
    __shared__ float maxima[8];
    __shared__ float scale;
    const float * x = input + size_t(blockIdx.x)*stride;
    float amax = 0;
    for (int i = threadIdx.x; i < k; i += blockDim.x) amax = fmaxf(amax, fabsf(x[i]));
    amax = block_reduce_max(amax, maxima);
    if (threadIdx.x == 0) {
        scale = __fdiv_rn(amax, 448.0f);
        scales[blockIdx.x] = scale == 0 ? 1.0f : scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        output[size_t(blockIdx.x)*k+i] = scale == 0 ? 0 :
            __nv_cvt_float_to_fp8(__fdiv_rn(x[i], scale), __NV_SATFINITE, __NV_E4M3);
    }
}

static __global__ void f8_lt_scale(const char * weight, const float * scales, float * output, int m, int n, int k, int ldc) {
    const int row = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    if (row >= m) return;
    float ws[4];
    for (int j = 0; j < 4; ++j) ws[j] = ggml_cuda_f8_row::scale(weight + size_t(row+j)*ggml_cuda_f8_row::row_size(k));
    for (int col = blockIdx.y*16; col < min(n, int(blockIdx.y+1)*16); ++col) {
        const size_t offset = size_t(col)*ldc + row;
        float4 values = *(const float4 *)(output + offset);
        float * v = (float *)&values;
        const float as = scales[col];
        for (int j = 0; j < 4; ++j) v[j] *= __fmul_rn(ws[j], as);
        *(float4 *)(output + offset) = values;
    }
}

struct ggml_cuda_f8_lt_context {
    struct plan {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t a, b, d;
        cublasLtMatmulAlgo_t algo;
        bool supported = false;
        ~plan() {
            CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b));
            CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d));
        }
    };
    static constexpr size_t workspace_size = 32*1024*1024;
    int device;
    cublasLtHandle_t handle;
    void * workspace;
    std::map<std::array<int,4>, std::unique_ptr<plan>> plans;

    explicit ggml_cuda_f8_lt_context(int device) : device(device) {
        ggml_cuda_set_device(device);
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    ~ggml_cuda_f8_lt_context() {
        ggml_cuda_set_device(device);
        plans.clear();
        CUDA_CHECK(cudaFree(workspace));
        CUBLAS_CHECK(cublasLtDestroy(handle));
    }
    plan & get(int m, int k, int n, int ldc) {
        const std::array<int,4> shape{m,k,n,ldc};
        auto found = plans.find(shape);
        if (found != plans.end()) return *found->second;
        auto p = std::make_unique<plan>();
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t trans = CUBLAS_OP_T;
        const int8_t fast = 0;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8F_E4M3, k, m, ggml_cuda_f8_row::row_size(k)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->b, CUDA_R_8F_E4M3, k, n, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->d, CUDA_R_32F, m, n, ldc));
        cublasLtMatmulPreference_t preference;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_size, sizeof(workspace_size)));
        const uint32_t alignment = 16;
        for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, attr, &alignment, sizeof(alignment)));
        }
        cublasLtMatmulHeuristicResult_t results[32];
        int count = 0, selected = -1;
        auto status = cublasLtMatmulAlgoGetHeuristic(handle, p->op, p->a, p->b, p->d, p->d, preference, 32, results, &count);
        if (status != CUBLAS_STATUS_NOT_SUPPORTED) CUBLAS_CHECK(status);
        auto attribute = [](const cublasLtMatmulAlgo_t & algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
            int value;
            size_t size;
            CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &size));
            return value;
        };
        if (status == CUBLAS_STATUS_SUCCESS && count) {
            const int split = attribute(results[0].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            bool preferred = false;
            for (int i = 0; i < count; ++i) {
                if (results[i].state != CUBLAS_STATUS_SUCCESS || attribute(results[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) != split) continue;
                cublasLtMatmulHeuristicResult_t checked{};
                auto check = cublasLtMatmulAlgoCheck(handle, p->op, p->a, p->b, p->d, p->d, &results[i].algo, &checked);
                if (check != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > workspace_size) continue;
                results[i].workspaceSize = checked.workspaceSize;
                const bool candidate_preferred = ggml_cuda_info().devices[device].cc == CC_ADA_LOVELACE &&
                    attribute(results[i].algo, CUBLASLT_ALGO_CONFIG_ID) == 35 &&
                    attribute(results[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID) == CUBLASLT_MATMUL_TILE_128x128 &&
                    attribute(results[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID) == CUBLASLT_MATMUL_STAGES_64x3;
                if (selected < 0 || candidate_preferred > preferred ||
                        (candidate_preferred == preferred && results[i].workspaceSize < results[selected].workspaceSize)) {
                    selected = i;
                    preferred = candidate_preferred;
                }
            }
        }
        p->supported = selected >= 0;
        if (p->supported) p->algo = results[selected].algo;
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        return *plans.emplace(shape, std::move(p)).first->second;
    }
    bool launch(int m, int k, int n, int ldc, const char * weight, const uint8_t * activation,
            const float * scales, float * output, cudaStream_t stream) {
        auto & p = get(m,k,n,ldc);
        if (!p.supported) return false;
        const float alpha = 1.0f, beta = 0;
        auto status = cublasLtMatmul(handle, p.op, &alpha, weight + ggml_cuda_f8_row::row_meta_size, p.a,
                activation, p.b, &beta, output, p.d, output, p.d, &p.algo, workspace, workspace_size, stream);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
            p.supported = false;
            return false;
        }
        CUBLAS_CHECK(status);
        f8_lt_scale<<<dim3((m+1023)/1024,(n+15)/16),256,0,stream>>>(weight,scales,output,m,n,k,ldc);
        return true;
    }
};

static ggml_cuda_f8_lt_context & f8_lt_context(ggml_backend_cuda_context & ctx) {
    if (!ctx.f8_lt) ctx.f8_lt = std::make_unique<ggml_cuda_f8_lt_context>(ctx.device);
    return *ctx.f8_lt;
}

static bool f8_lt_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    return f8_lt_eligible(node, ggml_cuda_info().devices[ctx.device].cc) &&
        f8_lt_context(ctx).get(node->ne[0],node->src[0]->ne[0],node->ne[1],node->ne[0]).supported;
}

static void f8_lt_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) f8_lt_supported(ctx, graph->nodes[i]);
}
#else
static bool f8_lt_supported(ggml_backend_cuda_context &, const ggml_tensor *) { return false; }
#endif
