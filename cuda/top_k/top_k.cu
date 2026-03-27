#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 定义 K 值（编译时常量，方便编译器优化寄存器分配）
#define K 10
#define BLOCK_SIZE 256

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d -%s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 核心逻辑：向一个有序数组中插入新元素 ---
__device__ __forceinline__ void insert_topk(float* vals, int* idxs, float new_val, int new_idx) {
    if (new_val <= vals[K- 1]) return;

    int i = K - 2;
    while (i >= 0 && vals[i] < new_val) {
        vals[i + 1] = vals[i];
        idxs[i + 1] = idxs[i];
        i--;
    }
    vals[i + 1] = new_val;
    idxs[i + 1] = new_idx;
}

// --- 第一阶段：每个 Block 寻找局部 Top-K ---
__global__ void topk_stage1(const float* input, int n, float* block_out_vals, int* block_out_idxs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // 1. 线程局部 Top-K (存储在寄存器中)
    float local_vals[K];
    int local_idxs[K];

    #pragma unroll
    for(int i = 0; i < K; i++) {
        local_vals[i] = -1e38f; // 极小值
        local_idxs[i] = -1;
    }

    // 2. 遍历分配给该线程的所有数据 (Grid-stride loop)
    for (int i = tid; i < n; i+= stride) {
        insert_topk(local_vals, local_idxs, input[i], i);
    }

    // 3. Block 内规约：使用 Shared Memory 合并所有线程的结果
    __shared__ float s_vals[BLOCK_SIZE * K];
    __shared__ int s_idxs[BLOCK_SIZE * K];

    // 将线程局部结果写入共享内存
    for (int i = 0; i < K; i++) {
        s_vals[tid * K + i] = local_vals[i];
        s_idxs[tid * K + i] = local_idxs[i];
    }
    __syncthreads();

    // 由 Block 内的前几个线程进行简单的合并（对于 K 较小的情况非常快）
    if (threadIdx.x == 0) {
        float final_vals[K];
        int final_idxs[K];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            final_vals[i] = -1e38f;
            final_idxs[i] =-1;
        }

        // 遍历共享内存中所有线程的 Top-K 结果
        for (int t = 0; t < BLOCK_SIZE; t++) {
            for (int i = 0; i < K; i++) {
                insert_topk(final_vals, final_idxs, s_vals[t * K + i], s_idxs[t * K + i]);
            }
        }

        // 将该 Block 的 Top-K 写入全局内存
        for (int i = 0; i < K; i++) {
            block_out_vals[blockIdx.x * K + i] = final_vals[i];
            block_out_idxs[blockIdx.x * K + i] = finalidxs[i];
        }
    }
}

// --- 第二阶段：汇总所有 Block 的结果得出全局 Top-K ---
__global__ void topk_stage2(const float* block_vals, const int* block_idxs, int num_blocks, float* final_vals, int* final_idxs) {
    // 仅由一个 Block 执行
    __shared__ float s_vals[BLOCK_SIZE * K];
    __shared__ int s_idxs[BLOCK_SIZE * K];

    // 1. 每个线程负责汇总一部分 Block 的结果
    float local_vals[K];
    int local_idxs[K];
    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -1e38f;
        local_idxs[i] = -1;
    }

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        for (int j = 0; j < K; j++) {
           insert_topk(local_vals, local_idxs, block_vals[i * K + j],block_idxs[i * K + j]);
        }
    }

    // 2. 再次在 Shared Memory 中规约
    for (int i = 0; i < K; i++) {
        s_vals[threadIdx.x * K + i] = local_vals[i];
        s_idxs[threadIdx.x * K + i] = local_idxs[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float res_vals[K];
        int res_idxs[K];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            res_vals[i] = -1e38f;
            res_idxs[i] = -1;
        }

        for (int t = 0; t < BLOCK_SIZE; t++) {
            for (int i = 0; i< K; i++) {
                insert_topk(res_vals, res_idxs, s_vals[t * K + i], s_idxs[t * K + i]);
            }
        }

        // 输出最终结果
        for (int i = 0; i < K; i++) {
            final_vals[i] = res_vals[i];
            final_idxs[i] = res_idxs[i];
        }
    }
}

int main() {
    const int N = 1000000; // 100万个数据
    const int num_blocks = 128;

    // Host 数据
    std::vector<float> h_input(N);
    for (int i = 0; i < N; i++) h_input[i] = (float)rand() / RAND_MAX;
    h_input[500] = 100.5f;   // 植入目标值
    h_input[9999] = 500.2f;
    h_input[888888] = 1000.1f;

    // Device 内存分配
    float *d_input, *d_block_vals, *d_final_vals;
    int *d_block_idxs, *d_final_idxs;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_vals, num_blocks* K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_idxs, num_blocks * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final_vals, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_idxs, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 执行两阶段 Top-K
    topk_stage1<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_block_vals, d_block_idxs);
    topk_stage2<<<1, BLOCK_SIZE>>>(d_block_vals, d_block_idxs, num_blocks, d_final_vals, d_final_idxs);

    // 拷贝结果回 Host
    float h_res_vals[K];
    int h_res_idxs[K];
    CUDA_CHECK(cudaMemcpy(h_res_vals, d_final_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_idxs, d_final_idxs, K * sizeof(int), cudaMemcpyDeviceToHost));

    // 打印结果
    printf("Top-%d Results:\n", K);
    for (int i = 0; i < K; i++) {
        printf("Rank %d: Value %f, Index %d\n", i + 1, h_res_vals[i], h_res_idxs[i]);
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
    cudaFree(d_final_vals);
    cudaFree(d_final_idxs);

    return 0;
}
