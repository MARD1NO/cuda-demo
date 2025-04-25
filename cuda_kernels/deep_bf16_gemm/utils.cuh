#pragma once 

#include <exception> 


namespace deep_gemm {


class AssertionException : public std::exception {
private:
    std::string message{};

public:
    explicit AssertionException(const std::string& message) : message(message) {}

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond)                                        \
do {                                                                \
    if (not (cond)) {                                               \
        printf("Assertion failed: %s:%d, condition: %s\n",          \
                __FILE__, __LINE__, #cond);                          \
        throw AssertionException("Assertion failed: " #cond);       \
    }                                                               \
} while (0)
#endif 


#ifndef DG_HOST_CHECK
#define DG_HOST_CHECK(status)                                        \
do {                                                                \
    if (status != cudaSuccess) {                                               \
        printf("Assertion failed: %s:%d, status: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(status));                          \
        throw AssertionException("CUDA CHECK FAILED");       \
    }                                                               \
} while (0)
#endif 


#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                          \
do {                                                                                    \
    if (not (cond)) {                                                                   \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);  \
        asm("trap;");                                                                   \
    }                                                                                   \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif


__device__ inline constexpr int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b; 
}

}  // namespace deep_gemm
