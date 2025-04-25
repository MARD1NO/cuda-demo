#include "bf16_gemm.cuh"

#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace deep_gemm {

template <typename T>
static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m, uint32_t block_k) {
    return make_2d_tma_desc(global_address, Layout::RowMajor,
                            shape_m * 1, shape_k, block_m, block_k);
}


template <typename T>
static CUtensorMap make_2d_tma_b_desc(T* global_address, uint32_t shape_k, uint32_t shape_n, uint32_t block_k, uint32_t block_n) {
    return make_2d_tma_desc(global_address, Layout::ColMajor,
                            shape_k, shape_n, block_k, block_n);
}

template <typename T>
static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m, uint32_t shape_n, uint32_t block_m, uint32_t block_n) {
    return make_2d_tma_desc(global_address, Layout::RowMajor,
                            shape_m * 1, shape_n, block_m, block_n,
                            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
static CUtensorMap make_2d_tma_desc(
        T* global_address, Layout layout,
        uint32_t gmem_rows, uint32_t gmem_cols,
        uint32_t smem_rows, uint32_t smem_cols,
        CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
    if (layout == Layout::RowMajor) {
        uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
        uint32_t smem_dim[2] = {smem_cols, smem_rows};
        return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
    } else {
        uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
        uint32_t smem_dim[2] = {smem_rows, smem_cols};
        return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
    }
}

void run_kernel(__nv_bfloat16* x, __nv_bfloat16* weight, __nv_bfloat16* y, 
                cudaStream_t stream) {
    constexpr int32_t BLOCK_M = 64; 
    constexpr int32_t BLOCK_K = 64; 
    constexpr int32_t BLOCK_N = 256; 

    constexpr int32_t M = 1024; 
    constexpr int32_t N = 1024; 
    constexpr int32_t K = 1024; 


    constexpr int32_t PER_STAGE_SMEM_A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16); 
    constexpr int32_t PER_STAGE_SMEM_B_SIZE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16); 
    constexpr int32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16); 
    constexpr int32_t PER_STAGE_BARRIER_SIZE = 2 * sizeof(int64_t); 

    constexpr int32_t kNumStages = 3; 

    size_t smem_barrier_size = kNumStages * sizeof(int64_t) * 2; 
    size_t smem_buf_size = kNumStages * (PER_STAGE_SMEM_A_SIZE + PER_STAGE_SMEM_B_SIZE) + SMEM_D_SIZE; 
    size_t smem_size = smem_buf_size + smem_barrier_size; 

    smem_size = (smem_size + 1024 - 1) / 1024 * 1024; 

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    constexpr uint32_t kNumGroups = 1;
    constexpr uint32_t kBlockSize = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M); 
    constexpr uint32_t kNumTMAMulticast = 1; 


    auto kernel = bf16_gemm_kernel<N, K, BLOCK_M, BLOCK_N, BLOCK_K, 
                                   kNumGroups, kNumStages, 
                                   kNumTMAThreads, kNumMathThreadsPerGroup, kNumTMAMulticast>;

    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); 

    auto tma_A_desc = make_2d_tma_a_desc(x, M, K, BLOCK_M, BLOCK_K); 
    auto tma_B_desc = make_2d_tma_b_desc(weight, K, N, BLOCK_K, BLOCK_N); 
    auto tma_D_desc = make_2d_tma_d_desc(y, M, N, BLOCK_M, BLOCK_N); 
    

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = 132;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;
    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel,
                                     M, 
                                     tma_A_desc, tma_B_desc, tma_D_desc);
    // DG_HOST_ASSERT(status == cudaSuccess);
    DG_HOST_CHECK(status);
}


void deep_gemm_bf16(const torch::Tensor& x, const torch::Tensor& weight, torch::Tensor& y) {
    auto m = x.size(0); 
    auto k = x.size(1); 
    auto n = weight.size(0); 


    TORCH_CHECK(m == 1024); 
    TORCH_CHECK(n == 1024); 
    TORCH_CHECK(k == 1024); 

    run_kernel(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()), 
               reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()), 
               reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), 
               at::cuda::getCurrentCUDAStream());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "deep_gemm_bf16",
        &deep_gemm_bf16,
        "deep_gemm_bf16");
}


} // deep_gemm 
