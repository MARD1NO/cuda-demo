

namespace tma_copy {

// template<int32_t BLOCK_M, int32_t BLOCK_N> 
// struct Scheduler {

//     int current_iter = -1; 
//     uint32_t num_m_blocks = -1; 
//     uint32_t num_n_blocks = -1; 
//     uint32_t num_blocks = -1; 

//     __device__ void Scheduler(uint32_t m, uint32_t n) {
//         num_m_blocks = (m + BLOCK_M - 1) / BLOCK_M; 
//         num_n_blocks = (n + BLOCK_N - 1) / BLOCK_N; 
//         num_blocks = num_m_blocks * num_n_blocks;
//     }

//     __device__ void get_block_idx(uint32_t next_block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
//         m_block_idx = next_block_idx / num_n_blocks;
//         n_block_idx = next_block_idx % num_n_blocks;
//     }

//     __device__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
//         uint32_t next_block_idx = (++current_iter) * gridDim.x + blockIdx.x; 
//         if next_block_idx >= num_blocks {
//             return false; 
//         } 
//         get_block_idx(next_block_idx, m_block_idx, n_block_idx);
//         return true; 
//     }

// }


template<int32_t BLOCK_M> 
struct Scheduler {

    int current_iter = -1; 
    int32_t num_blocks = -1; 

    __device__ Scheduler(uint32_t m) {
        uint32_t num_m_blocks = (m + BLOCK_M - 1) / BLOCK_M; 
        num_blocks = num_m_blocks;
    }

    __device__ bool get_next_block(uint32_t& m_block_idx) {
        uint32_t next_block_idx = (++current_iter) * gridDim.x + blockIdx.x; 
        if (next_block_idx >= num_blocks) {
            return false; 
        } 
        m_block_idx = next_block_idx; 
        return true; 
    }
}; 



} // tma_copy