#include "cuda_bf16.h"
#include "stdint.h"
#include "stdio.h"


__device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr)
{
    // return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    uint32_t smem_ptr;

    asm(
    "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr) : "l"(ptr));

    return smem_ptr;

// // We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// // the previous internal intrinsics if they are available.
// #if CUTE_CVTA_GENERIC_TO_SHARED_ACTIVATED
//   //
//   // This NVVM intrinsic converts an address in shared memory to a plain
//   // unsigned integer. This is necessary to pass to shared memory instructions
//   // in inline PTX.
//   //
//   // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
//   //
//   //__device__ size_t __cvta_generic_to_shared(void* ptr);

//   /// CUTE helper to get SMEM pointer
//   return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

// #elif CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED

//   return __nvvm_get_smem_pointer(ptr);

// #elif defined(__CUDA_ARCH__)

//   uint32_t smem_ptr;

//   asm(
//   "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
//     : "=r"(smem_ptr) : "l"(ptr));

//   return smem_ptr;

// #else


//   (void) ptr;
//   printf("ERROR: cast_smem_ptr_to_uint not supported but used.\n");
//   return 0;

// #endif
}

// __global__ void test_ld_matrix() {
//     __nv_bfloat16 reg[2]; 


//     /*
//     T0: 0, 1
//     T1: 2, 3
//     */
//     for (int i = 0; i < 2; i++) {
//         reg[i] = 2 * threadIdx.x + i; 
//     }

//     printf("reg[0] = %f\n", static_cast<float>(reg[0]));
//     printf("reg[1] = %f\n", static_cast<float>(reg[1]));
//     __shared__ __nv_bfloat16 smem[8 * 8]; 

//     uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem);
//     uint32_t dst = *reinterpret_cast<uint32_t*>(reg);

//     asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
//         : "=r"(dst)
//         :  "r"(smem_int_ptr));

//     __syncwarp(); 

//     if (threadIdx.x == 0) {
//         for (int row = 0; row < 8; row++) {
//             for (int col = 0; col < 8; col++) {
//                 printf("smem[%d][%d] = %f\n", row, col, static_cast<float>(smem[row * 8 + col]));
//             }
//         }
//     }
// }



template <typename dtype_t>
struct SM90_U32x4_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst) {
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
                                 *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

__device__ __nv_bfloat162 pack_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat162{a, b};
}


__global__ void test_st_matrix() {
    __nv_bfloat16 reg[8]; 
    const int32_t warp_idx = threadIdx.x / 32; 
    const int32_t lane_idx = threadIdx.x % 32; 
    const int32_t BLOCK_N = 16; 
    const int32_t i = 0; 

    for (int i = 0; i < 8; i++) {
        reg[i] = 8 * threadIdx.x + i; 
    }

    __shared__ __nv_bfloat16 smem[16 * 16];
    if (lane_idx == 0) {
        for (int i = 0; i < 8 * 8; i++) {
            smem[i] = 0; 
        }
    }

    __syncthreads(); 

    auto smem_d = smem + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16); 

    const int32_t address = (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16); 
    const int32_t address_row = address / 16; 
    const int32_t address_col = address % 16; 

    printf("threadIdx.x = %d, address = %d, address_row = %d, address_col = %d\n", threadIdx.x, address, address_row, address_col);

    SM90_U32x4_STSM_N<__nv_bfloat162>::copy(pack_bfloat162(reg[0], reg[1]), pack_bfloat162(reg[2], reg[3]), 
                                            pack_bfloat162(reg[4], reg[5]), pack_bfloat162(reg[6], reg[7]), smem_d);

    if (threadIdx.x == 0) {
        for (int row = 0; row < 16; row++) {
            printf("row %d:", row); 
            for (int col = 0; col < 16; col++) {
                // printf("smem[%d][%d] = %f\n", row, col, static_cast<float>(smem[row * 8 + col]));
                printf("%.1f, ", static_cast<float>(smem[row * 16 + col]));
            }
            printf("\n");
        }
    }                       
}

int main() {
    test_st_matrix<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}