#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor, 
    ColMajor
}; 

template<uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "kNumMathThreadsPerGroup should be 128"); 
    return (block_m == 64 ? 1: 2) * kNumMathThreadsPerGroup + kNumTMAThreads; 
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K, 
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, 
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
bf16_gemm_kernel(
    uint32_t shape_m, 
    const __grid_constant__ CUtensorMap tensor_map_a, 
    const __grid_constant__ CUtensorMap tensor_map_b, 
    const __grid_constant__ CUtensorMap tensor_map_d
) {

    using Barrier = cutlass::arch::ClusterTransactionBarrier; 

    // Shared Memory 
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_K * BLOCK_N * sizeof(__nv_bfloat16);

    // Configs 
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K; 
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages); 
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0); 
    const uint32_t lane_idx = threadIdx.x % 32;

    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_d));
    }
    __syncwarp(); 

    extern __shared__ __align__(1024) uint8_t smem_buffer[]; 
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_bfloat16* smem_a[kNumStages]; 
    __nv_bfloat16* smem_b[kNumStages]; 

    // TMA Barrier
    Barrier* full_barriers[kNumStages]; 
    Barrier* empty_barriers[kNumStages]; 

    #pragma unroll 
    for(int i = 0; i < kNumStages; i++) {
        smem_a[i] = reinterpret_cast<__nv_bfloat16*>(smem_buffer + SMEM_D_SIZE + 
                                                     i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_bfloat16*>(smem_buffer + SMEM_D_SIZE + 
                                                     kNumStages * SMEM_A_SIZE_PER_STAGE + 
                                                     i * SMEM_B_SIZE_PER_STAGE);
    }

    // Fill baarriers 
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_D_SIZE + 
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
    ); 

    #pragma unroll
    for(int i = 0; i < kNumStages; i++) {
        full_barriers[i] = barrier_start_ptr + i; 
        empty_barriers[i] = barrier_start_ptr + kNumStages + i; 
    }

    // Initialize barriers 
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll 
        for(int i = 0; i < kNumStages; i++) {
            full_barriers[i]->init(1); 
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32); 
        }

        cutlass::arch::fence_view_async_shared(); 
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling 
    struct DivisibleK{}; 
    struct NotDivisibleK{};
    auto launch_k_iterations = [](const auto& func) {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; k_iter++) {
                func(k_iter, DivisibleK{}); 
            }
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; k_iter++) {
                func(k_iter, DivisibleK{}); 
            }
            func(kNumIterations - 1, NotDivisibleK{}); 
        }
    }; 

    constexpr int kNumTMARegisters = 40; 
    constexpr int kNumMathRegisters = 232;

    // BlockScheduler 
    uint32_t m_block_idx, n_block_idx; 
    auto scheduler = Scheduler<GemmType::Normal, 
                               SHAPE_N, BLOCK_M, BLOCK_N,
                               kNumGroups, kNumTMAMulticast>(shape_m);

    if (threadIdx.x >= kNumMathThreads) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>(); 

        if (threadIdx.x == kNumMathThreads) {
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K; 

                    #pragma unroll 
                    for (int32_t s = 0; s < kNumInnerStages; ++s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1); 
                        
                        auto& full_barrier = *full_barriers[s]; 
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K; 
                        
                        tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_a[s], k_idx, scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                        
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));

                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE); 
                    }

                    #pragma unroll 
                    for(int32_t s = kNumInnerStages; s < kNumStages; s++) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1); 
                        full_barriers[s]->arrive(); 
                    }
                }); 
            }
        }
    } else {
        // Math Warp group
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>(); 
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {

            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulate 
            float accum[WGMMA::kNumAccum] = {0}; 
            
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            }; 

            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K; 

                // if (threadIdx.x == 0 && blockIdx.x == 0) {
                //     printf("kNumInnerStages: %d. \n", kNumInnerStages);
                // }

                #pragma unroll 
                for (int32_t s = 0; s < kNumInnerStages; ++s) {
                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    // commit wgmma instructions 
                    #pragma unroll 
                    for (int i = 0; i < WGMMA::kNumAccum; i++) {
                        warpgroup_fence_operand(accum[i]); 
                    }
                    warpgroup_arrive(); 

                    #pragma unroll 
                    for (int k = 0; k < BLOCK_K / WGMMA::K; k++) {
                        auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);

                        
                        // WGMMA::wgmma(desc_a, desc_b, accum, k);
                        // Note(Zhengzekang): Here we clear the accumulator, so we pass 1(True) to let gmma do accumulate instead of overwrite. 
                        WGMMA::wgmma(desc_a, desc_b, accum, 1);
                    }
                    warpgroup_commit_batch(); 
                    #pragma unroll 
                    for (int i = 0; i < WGMMA::kNumAccum; i++) {
                        warpgroup_fence_operand(accum[i]); 
                    }
                    warpgroup_wait<0>(); 

                    // Notify barrier arrival
                    empty_barrier_arrive(s);
                }

                #pragma unroll 
                for(int32_t s = kNumInnerStages; s < kNumStages; s++) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1); 
                    empty_barrier_arrive(s);
                }
            }); 
            
            #pragma unroll
            for(auto i = 0; i < WGMMA::kNumAccum / 8; ++i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({accum[i * 8 + 0], accum[i * 8 + 1]}),
                    __float22bfloat162_rn({accum[i * 8 + 2], accum[i * 8 + 3]}),
                    __float22bfloat162_rn({accum[i * 8 + 4], accum[i * 8 + 5]}),
                    __float22bfloat162_rn({accum[i * 8 + 6], accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            cute::tma_store_fence(); 
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            if (threadIdx.x == 0) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N,
                                              scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncwarp();
        }
    }
}

} // namespace deep_gemm 

