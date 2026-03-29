#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// 定义 K 值和 Block 大小
#define K 10
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 辅助函数：将单个元素插入有序数组 ---
__device__ __forceinline__ void insert_element(float* vals, int* idxs, float val, int idx) {
    if (val <= vals[K - 1]) return;
    int i = K - 2;
    while (i >= 0 && vals[i] < val) {
        vals[i + 1] = vals[i];
        idxs[i + 1] = idxs[i];
        i--;
    }
    vals[i + 1] = val;
    idxs[i + 1] = idx;
}

// --- 核心优化：合并两个有序的 Top-K 序列 ---
__device__ __forceinline__ void merge_sequences(float* my_vals, int* my_idxs, const float* other_vals, const int* other_idxs) {
    float temp_vals[K];
    int temp_idxs[K];
    int i = 0, j = 0;

    #pragma unroll
    for (int count = 0; count < K; count++) {
        if (my_vals[i] >= other_vals[j]) {
            temp_vals[count] = my_vals[i];
            temp_idxs[count] = my_idxs[i];
            i++;
        } else {
            temp_vals[count] = other_vals[j];
            temp_idxs[count] = other_idxs[j];
            j++;
        }
    }
    #pragma unroll
    for (int count = 0; count < K; count++) {
        my_vals[count] = temp_vals[count];
        my_idxs[count] = temp_idxs[count];
    }
}

// --- Warp 级别规约：使用 Shuffle 指令 ---
__device__ __forceinline__ void warp_reduce_topk(float* vals, int* idxs) {
    float remote_vals[K];
    int remote_idxs[K];

    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            // 直接从其他线程的寄存器拉取数据
            remote_vals[i] = __shfl_down_sync(FULL_MASK, vals[i], delta);
            remote_idxs[i] = __shfl_down_sync(FULL_MASK, idxs[i], delta);
        }
        // 合并两个有序序列
        merge_sequences(vals, idxs, remote_vals, remote_idxs);
    }
}

// --- 第一阶段：Block 级 Top-K (含 Warp Shuffle 优化) ---
__global__ void topk_stage1_shuffle(const float* input, int n, float* block_out_vals, int* block_out_idxs) {
    float local_vals[K];
    int local_idxs[K];

    #pragma unroll
    for(int i = 0; i < K; i++) {
        local_vals[i] = -1e38f;
        local_idxs[i] = -1;
    }

    // 1. 线程级处理：Grid-stride loop
    for (int i = blockIdx.x *blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        insert_element(local_vals, local_idxs, input[i], i);
    }

    // 2. Warp 级规约：每个 Warp 内部 32 线程合并
    warp_reduce_topk(local_vals, local_idxs);

    // 3. Block 级规约：Warp 间合并
    __shared__ float s_vals[K * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ int s_idxs[K * (BLOCK_SIZE / WARP_SIZE)];

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 每个 Warp 的第一个线程将结果存入共享内存
    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            s_vals[warp_id * K + i] = local_vals[i];
            s_idxs[warp_id * K + i] = local_idxs[i];
        }
    }
    __syncthreads();

    // 4. 最后由第一个 Warp 合并共享内存中的所有 Warp 结果
    if (warp_id == 0) {
        float final_vals[K];
        int final_idxs[K];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            final_vals[i] = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? s_vals[lane_id * K + i] : -1e38f;
            final_idxs[i] = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? s_idxs[lane_id * K + i] : -1;
        }

        warp_reduce_topk(final_vals, final_idxs);

        if (lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                block_out_vals[blockIdx.x* K + i] = final_vals[i];
                block_out_idxs[blockIdx.x * K + i] = final_idxs[i];
            }
        }
    }
}

// --- 第二阶段：全局汇总 (单 Block 执行) ---
__global__ void topk_stage2_shuffle(const float* block_vals, const int* block_idxs, int num_blocks, float* final_vals, int* final_idxs) {
    float local_vals[K];
    int local_idxs[K];

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_vals[i] = -1e38f;
        local_idxs[i] = -1;
    }

    // 汇总所有 Block 的结果
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        for (int j = 0; j< K; j++) {
            insert_element(local_vals, local_idxs, block_vals[i * K + j], block_idxs[i * K + j]);
        }
    }

    // 同样使用 Warp Shuffle 进行 Block 内规约
    warp_reduce_topk(local_vals, local_idxs);

    __shared__ float s_vals[K * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ int s_idxs[K * (BLOCK_SIZE / WARP_SIZE)];

    intlane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
        for (int i = 0; i < K; i++) {
            s_vals[warp_id * K + i] = local_vals[i];
            s_idxs[warp_id * K + i] = local_idxs[i];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float res_vals[K];
        int res_idxs[K];
        #pragma unroll
        for (int i = 0; i < K; i++) {
            res_vals[i] = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? s_vals[lane_id * K + i] : -1e38f;
            res_idxs[i] = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? s_idxs[lane_id * K + i] : -1;
        }
        warp_reduce_topk(resvals, res_idxs);
        if (lane_id == 0) {
            for (int i = 0; i < K; i++) {
                final_vals[i] = res_vals[i];
                final_idxs[i] = res_idxs[i];
            }
        }
    }
}

int main() {
    const int N = 1000000;
    const int num_blocks = 256;

    std::vector<float> h_input(N);
    for (int i = 0; i < N; i++) h_input[i] = (float)rand() / RAND_MAX;
    h_input[123] = 999.9f;
    h_input[8888] = 1000.5f;
    h_input[999999] = 888.8f;

    float *d_input, *d_block_vals, *d_final_vals;
    int *d_block_idxs, *d_final_idxs;

    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_vals, num_blocks* K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_idxs, num_blocks * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final_vals, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_idxs, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 执行
    topk_stage1_shuffle<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_block_vals, d_block_idxs);
    topk_stage2_shuffle<<<1, BLOCK_SIZE>>>(d_block_vals, d_block_idxs, num_blocks, d_final_vals, d_final_idxs);

    float h_res_vals[K];
    int h_res_idxs[K];
    CUDA_CHECK(cudaMemcpy(h_res_vals, d_final_vals, K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_res_idxs, d_final_idxs, K * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Top-%d Results (Optimized with Warp Shuffle):\n", K);
    for (int i = 0; i < K; i++) {
        printf("Rank %d: Value %f, Index %d\n",i + 1, h_res_vals[i], h_res_idxs[i]);
    }

    cudaFree(d_input); cudaFree(d_block_vals); cudaFree(d_block_idxs);
    cudaFree(d_final_vals); cudaFree(d_final_idxs);
    return 0;
}
