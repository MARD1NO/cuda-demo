#include "utils.cuh"


namespace deep_gemm {

enum class GemmType {
    Normal
}; 


template <GemmType kGemmType, 
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct Scheduler {
    int current_iter = -1; 
    uint32_t num_aligned_m_blocks; 

    uint32_t num_blocks; 

    __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M); 
        num_blocks = num_aligned_m_blocks * kNumNBlocks;
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, 
                                                           int block_idx, 
                                                           uint32_t& m_block_idx,
                                                           uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");
        auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup; 
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
        auto in_group_idx = block_idx % num_blocks_per_group;
        m_block_idx = in_group_idx / num_n_blocks_in_group;
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } 
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x; 
        if (next_block_idx >= num_blocks) {
            return false; 
        } else {
            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx); 
            return true; 
        }
        return true; 
    }
}; 


}  // namespace deep_gemm
