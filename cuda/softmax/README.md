## Softmax

https://github.com/Starmys/TritonStudyGroup/blob/main/1_CUDA_Softmax/csrc/better_softmax.cu

### formula

$$Sofmax(x_i)= \frac{e^{x_i}}{\sum_j{e^{x_j}}}$$

### cuda naive impl

每个thread处理一行数据，对单行数据进行3-pass softmax处理

```c++
// CUDA kernel for naive softmax implementation
__global__ void naive_softmax_kernel(float* x, float* y, int batch_size, int hidden_dim) {
    // Each thread processes one row of the input matrix x
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary check
    if (row_idx >= batch_size) return;

    // Calculate the maximum value in the row
    float max_val = -FLT_MAX;
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        max_val = max(max_val, tmp_val);
    }

    // Calculate the sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        sum_exp += expf(tmp_val - max_val);
    }

    // Write the softmax output
    for (int i = 0; i < hidden_dim; i++) {
        float tmp_val = x[row_idx * hidden_dim + i];  // Read from global memory
        y[row_idx * hidden_dim + i] = expf(tmp_val - max_val) / sum_exp;  // Write to global memory
    }
}
```

### 优化1（ 1个warp处理1行， 向量化读取）

vals_per_thread = hidden_dim / WARP_SIZE, 确保小于255（一个thread最大255个register）

```cpp
#define WARP_SIZE 32          // Number of threads in a warp
#define WARP_MASK 0xFFFFFFFF  // Mask for all threads in a warp: 0xFFFFFFFF = 0b11111111111111111111111111111111
#define MEM_ACCESS_WIDTH 4    // Number of floats accessed in a single memory operation (4 floats = 16 bytes = 128 bits)


template<int VALS_PER_THREAD>  // Each thread (lane) processes VALS_PER_THREAD values
__global__ void better_softmax_kernel(float* x, float* y, int batch_size) {
    // Current warp index in the thread block
    const int warp_idx = threadIdx.x / WARP_SIZE;
    // Current thread lane index within the warp
    const int lane_idx = threadIdx.x % WARP_SIZE;
    // Number of warps in a thread block
    const int num_warps = blockDim.x / WARP_SIZE;

    // Each warp processes one row
    const int row_idx = blockIdx.x * num_warps + warp_idx;
    // Boundary check
    if (row_idx >= batch_size) return;

    // Offset for contiguous memory access in a warp
    const int offset = row_idx * (WARP_SIZE * VALS_PER_THREAD) + lane_idx * MEM_ACCESS_WIDTH;

    // Allocate VALS_PER_THREAD floats in registers
    float tmp_val[VALS_PER_THREAD];

    // Load VALS_PER_THREAD values from global memory into registers
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        // Vectorized memory access using float4
        reinterpret_cast<float4*>(&tmp_val[i])[0] =
            reinterpret_cast<float4*>(&x[offset + i * WARP_SIZE])[0];
    }

    // Find the maximum value in the thread's values
    float max_val = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        max_val = max(max_val, tmp_val[i]);
    }
    // Reduce the maximum value across all threads in the warp
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        max_val = max(max_val, __shfl_xor_sync(WARP_MASK, max_val, laneMask));
    }

    // Calculate exponential values and sum of exponentials in the thread's values
    float sum_exp = 0.0f;
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] = expf(tmp_val[i] - max_val);
        sum_exp += tmp_val[i];
    }
    // Reduce the sum of exponentials across all threads in the warp
    #pragma unroll
    for (int laneMask = 1; laneMask < WARP_SIZE; laneMask <<= 1) {
        sum_exp += __shfl_xor_sync(WARP_MASK, sum_exp, laneMask);
    }

    // Calculate the softmax values
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        tmp_val[i] /= sum_exp;
    }

    // Write VALS_PER_THREAD values registers to global memory 
    #pragma unroll
    for (int i = 0; i < VALS_PER_THREAD; i += MEM_ACCESS_WIDTH) {
        // Vectorized memory access using float4
        reinterpret_cast<float4*>(&y[offset + i * WARP_SIZE])[0] =
            reinterpret_cast<float4*>(&tmp_val[i])[0];
    }
}
```