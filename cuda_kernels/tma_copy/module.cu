#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>


#include "tma_utils.cuh"


#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace tma_copy {

enum class Layout {
    RowMajor, 
    ColumnMajor
}; 

template <typename T>
static CUtensorMap make_2d_tma_desc(
        T* global_address, Layout layout,
        uint32_t gmem_rows, uint32_t gmem_cols,
        uint32_t smem_rows, uint32_t smem_cols,
        CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
    // Need to flip the dim. 
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


template<int32_t BLOCK_M, int32_t BLOCK_K, int32_t kNumStages>
__global__ void tma_copy_kernel(const float* x, float* y, int32_t rows, int32_t cols, 
                                const __grid_constant__ CUtensorMap tensor_map, 
                                const __grid_constant__ CUtensorMap tensor_out_map) {

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_COPY_SIZE_PERSTAGE = BLOCK_M * BLOCK_K * sizeof(float);
    constexpr uint32_t SMEM_COPY_SIZE = kNumStages * SMEM_COPY_SIZE_PERSTAGE; 
    constexpr uint32_t kNumTMAMulticast = 1; 
    
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_COPY_SIZE);

    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    float* smem_a[kNumStages]; 
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<float*>(smem_buffer + i * SMEM_COPY_SIZE_PERSTAGE);
    }

    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_out_map));

    }
    __syncwarp();

    #pragma unroll 
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            // empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
            empty_barriers[i]->init(kNumTMAMulticast * 32 / 32);

        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFullKOfStages = BLOCK_K * kNumStages; 
    uint32_t num_iterations = cols / kFullKOfStages; 
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);


    if (warp_idx == 0) {
        if (threadIdx.x == 0) {
            // Warp 0 is responsible for copy data to smem. 
            for(int copy_iteration = 0; copy_iteration < num_iterations; ++copy_iteration) {
                for(int s = 0; s < kNumStages; s++) {
                    // empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                    empty_barriers[s]->wait((copy_iteration + 1) & 1);
                    auto& full_barrier = *full_barriers[s];
                    tma_copy<kNumTMAMulticast>(&tensor_map, reinterpret_cast<uint64_t*>(&full_barrier),
                                               smem_a[s], 
                                               copy_iteration * kNumStages * BLOCK_K + s * BLOCK_K, 
                                               0);
                    full_barrier.arrive_and_expect_tx(SMEM_COPY_SIZE_PERSTAGE);
                }
            }
        }
    } else {
        uint32_t worker_id = threadIdx.x % 32;

        // Warp 0 is responsible for copy data to smem. 
        // wait and arrive
        for(int copy_iteration = 0; copy_iteration < num_iterations; ++copy_iteration) {
            for(int s = 0; s < kNumStages; s++) {
                auto& full_barrier = *full_barriers[s];
                full_barriers[s]->wait((copy_iteration) & 1);
                
                uint32_t y_offset = copy_iteration * kNumStages * BLOCK_K + s * BLOCK_K; 
                
                // Use TMA store to write back to global memory
                if (worker_id == 0) {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_out_map, smem_a[s], copy_iteration * kNumStages * BLOCK_K + s * BLOCK_K, 0);
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                __syncwarp();
                
                if (worker_id == 0) {
                    empty_barriers[s]->arrive();
                }
            }
        }
    }
}


void run_kernel(const float* x, float* y, int32_t rows, int32_t cols, cudaStream_t stream) {
    constexpr int32_t BLOCK_M = 32; 
    constexpr int32_t BLOCK_K = 32; 
    constexpr int32_t PER_STAGE_COPY_SIZE = BLOCK_M * BLOCK_K * sizeof(float); 
    constexpr int32_t kNumStages = 2; 

    size_t smem_barrier_size = kNumStages * sizeof(int64_t) * 2; 
    size_t smem_buf_size = kNumStages * PER_STAGE_COPY_SIZE; 
    size_t smem_size = smem_barrier_size + smem_buf_size; 
    smem_size = (smem_size + 1024 - 1) / 1024 * 1024; 

    cudaFuncSetAttribute(tma_copy_kernel<BLOCK_M, BLOCK_K, kNumStages>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); 

    // auto tma_a_desc = make_2d_tma_desc(x, Layout::RowMajor, rows, cols, BLOCK_M, BLOCK_K, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    // auto tma_out_desc = make_2d_tma_desc(y, Layout::RowMajor, rows, cols, BLOCK_M, BLOCK_K, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);

    auto tma_a_desc = make_2d_tma_desc(x, Layout::RowMajor, rows, cols, BLOCK_M, BLOCK_K);
    auto tma_out_desc = make_2d_tma_desc(y, Layout::RowMajor, rows, cols, BLOCK_M, BLOCK_K);

    tma_copy_kernel<BLOCK_M, BLOCK_K, kNumStages><<<1, 64, smem_size, stream>>>(x, y, rows, cols, tma_a_desc, tma_out_desc);
}


void tma_copy_func(const torch::Tensor& x, torch::Tensor& y) {
  auto rows = x.size(0);
  auto cols = x.size(1);
  TORCH_CHECK(rows == 32); 
  TORCH_CHECK(cols % 64 == 0); 
  run_kernel(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols,
             at::cuda::getCurrentCUDAStream()); 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def(
//     "tma_copy",
//     &tma_copy,
//     "tma copy demo");

m.def(
    "tma_copy",
    &tma_copy_func,
    "cublas fp16 bf16 fp32 gemm");
}


} // tma_copy
