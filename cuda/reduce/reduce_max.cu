#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FLT_MAX ((float)(1e10))
#define WARP_MASK 0xFFFFFFFF
// #define RAND_MAX 1024

// https://zhuanlan.zhihu.com/p/1964020134839576011
// warp_reduce
template <typename T, int REDUCE_SIZE, template<typename> class OP>
__forceinline__ __device__ T warp_reduce(T val) {
    for (int i = REDUCE_SIZE / 2; i >= 1; i >>= 1) {
        val = OP<T>()(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

template <typename T>
struct Add {
    __device__ __forceinline__ T operator() (const T& x, const T&y) {
        return x + y;
    }
};

template <typename T>
struct Max {
    __device__ __forceinline__ T operator() (const T& x, const T&y) {
        return x > y ? x : y;
    }
};

// reduce basic impl
__global__ void reduce_basic(float *input, float *output, 
    int batch_size, int dim) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= batch_size) return;

    float *row_input = input + row_idx * dim;

    // 阶段 1: 查找最大值
    float max_val = row_input[0];
    for (int i = 1; i < dim; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    output[row_idx] = max_val;
}

// reduce (one block for one row)
__global__ void reduce_one_block_for_one_row(float *input, float *output, 
    int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];

    // 阶段 1: 并行 max 归约
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[batch_idx * dim + i]);
    }
    shared_mem[tid] = thread_max;
    __syncthreads();

    // 树形归约
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], 
                        shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_mem[0];
    output[batch_idx] = max_val;
}

// reduce (one warp for one row, using warp_reduce)
__global__ void reduce_one_warp_for_one_row(float *input, float *output, 
    int batch_size, int dim) {

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_num = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int cur_row = blockIdx.x * warp_num + warp_id;
    if (cur_row >= batch_size) return;
    int offset = cur_row * dim;

    float max_val = -FLT_MAX;
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        max_val = fmaxf(max_val, input[offset + i]);
    }
    max_val = warp_reduce<float, WARP_SIZE, Max>(max_val);
    output[cur_row] = max_val;
}


void benchmark_reduce(float *input, float *output, 
    int batch_size, int dim) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    reduce_one_warp_for_one_row<<<batch_size, 128>>>(input, output, batch_size, dim);
    // reduce_one_block_for_one_row<<<batch_size, 128>>>(input, output, batch_size, dim);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time = 0;
    cudaEventElapsedTime(&time, start, end);
    printf("reduce time: %f ms\n", time);
}

void benchmark_reduce_ref(float *input, float *output, 
    int batch_size, int dim) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    reduce_basic<<<batch_size, 128>>>(input, output, batch_size, dim);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time = 0;
    cudaEventElapsedTime(&time, start, end);
    printf("[ref]reduce time: %f ms\n", time);
}

int main() {
    int batch_size = 1024;
    int dim = 1024;
    float *input = new float[batch_size * dim];
    float *output = new float[batch_size];
    float *output_ref = new float[batch_size];
    for (int i = 0; i < batch_size * dim; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));
    cudaMemcpy(d_input, input, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 5; i ++) {
        benchmark_reduce(d_input, d_output, batch_size, dim);
    }
    cudaMemcpy(output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i ++) {
        benchmark_reduce_ref(d_input, d_output, batch_size, dim);
    }
    cudaMemcpy(output_ref, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    bool all_match = true;
    for(int i = 0; i < batch_size; i++) {
        if(i < 10) {
            printf("[idx-%d] GPU max = %.6f, CPU max = %.6f\n", i, output[i], output_ref[i]);
        }
        if(output[i] != output_ref[i]) {
            printf("Mismatch at index %d: GPU max = %.6f, CPU max = %.6f\n", i, output[i], output_ref[i]);
            all_match = false;
        }
    }
    if(all_match) {
        printf("All results match!\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input;
    delete[] output;
    return 0;
}