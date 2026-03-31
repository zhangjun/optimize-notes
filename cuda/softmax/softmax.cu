#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FLT_MAX ((float)(1e10))
#define WARP_MASK 0xFFFFFFFF

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

// softmax basic impl
__global__ void softmax_basic(float *input, float *output, 
    int batch_size, int dim) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= batch_size) return;

    float *row_input = input + row_idx * dim;
    float *row_output = output + row_idx * dim;

    // 阶段 1: 查找最大值
    float max_val = row_input[0];
    for (int i = 1; i < dim; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // 阶段 2: 计算指数和
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_exp += __expf(row_input[i] - max_val);
    }

    // 阶段 3: 归一化
    for (int i = 0; i < dim; i++) {
        row_output[i] = __expf(row_input[i] - max_val) / sum_exp;
    }
}

// softmax (one block for one row)
__global__ void softmax_one_block_for_one_row(float *input, float *output, 
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

    // 阶段 2: 并行 sum 归约（类似模式）
    float sum_exp = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum_exp += __expf(input[batch_idx * dim + i] - max_val);
    }
    shared_mem[tid] = sum_exp;
    __syncthreads();

    // 树形归约
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_mem[0];

    // 阶段 3: 并行归一化
    for (int i = tid; i < dim; i += blockDim.x) {
        output[batch_idx * dim + i] = __expf(input[batch_idx * dim + i] - max_val) / sum_exp;
    }
}

// softmax (one warp for one row)
__global__ void softmax_one_warp_for_one_row(float *input, float *output, 
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

    float sum_exp = 0.0f;
    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        sum_exp += __expf(input[offset + i] - max_val);
    }
    sum_exp = warp_reduce<float, WARP_SIZE, Add>(sum_exp);

    for (int i = lane_id; i < dim; i += WARP_SIZE) {
        output[offset + i] = __expf(input[offset + i] - max_val) / sum_exp;
    }
}

// Single-pass Online softmax kernel (most efficient)
__global__ void softmax_online(float *input, float *output, int batch_size, int dim) {
    auto block = this_thread_block();
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    extern __shared__ float shared_data[];
    
    // Online algorithm for numerical stable softmax
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // First pass: compute max and sum in a numerically stable way
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = row_input[i];
        float old_max = max_val;
        max_val = fmaxf(max_val, val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(val - max_val);
    }
    
    // Store thread-local values
    shared_data[tid] = max_val;
    shared_data[tid + blockDim.x] = sum_exp;
    block.sync();
    
    // Reduce max values
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other_max = shared_data[tid + stride];
            float other_sum = shared_data[tid + stride + blockDim.x];
            float new_max = fmaxf(max_val, other_max);
            sum_exp = sum_exp * expf(max_val - new_max) + 
                     other_sum * expf(other_max - new_max);
            max_val = new_max;
            shared_data[tid] = max_val;
            shared_data[tid + blockDim.x] = sum_exp;
        }
        block.sync();
    }
    
    float global_max = shared_data[0];
    float global_sum = shared_data[blockDim.x];
    
    // Second pass: compute final softmax values
    for (int i = tid; i < dim; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - global_max) / global_sum;
    }
}


#define WARP_SIZE 32
#define NEG_INF (-1e20f)

struct MD {
    float m;  // running max
    float d;  // running sum(exp(x - m))
};

__device__ __forceinline__ MD md_combine(const MD& a, const MD& b) {
    MD out;
    out.m = a.m > b.m ? a.m : b.m;
    out.d = a.d * expf(a.m - out.m) + b.d * expf(b.m - out.m);
    return out;
}

template <int threads_per_block = 128>
__global__ void online_softmax_kernel(float* input, float* output, int row, int col) {
    constexpr int warp_num = threads_per_block / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int cur_row = blockIdx.x * warp_num + warp_id;
    if (cur_row >= row) return;

    const float* x = input + cur_row * col;
    float* y = output + cur_row * col;

    // 1) 每个 lane 顺序扫描自己的列分片，得到局部 (m, d)
    MD local{NEG_INF, 0.f};
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        MD elem{x[i], 1.f};
        local = md_combine(local, elem);
    }

    // 2) warp 内归约，得到整行 (m, d)
    for (int mask = WARP_SIZE / 2; mask >= 1; mask >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, local.m, mask);
        other.d = __shfl_xor_sync(0xffffffff, local.d, mask);
        local = md_combine(local, other);
    }

    float row_m = local.m;
    float inv_d = 1.f / local.d;

    // 3) 写回 softmax
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        y[i] = expf(x[i] - row_m) * inv_d;
    }
}

void benchmark_softmax(float *input, float *output, 
    int batch_size, int dim) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    softmax_one_warp_for_one_row<<<batch_size, 128>>>(input, output, batch_size, dim);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time = 0;
    cudaEventElapsedTime(&time, start, end);
    printf("softmax time: %f ms\n", time);
}

int main() {
    int batch_size = 1024;
    int dim = 1024;
    float *input = new float[batch_size * dim];
    float *output = new float[batch_size * dim];
    for (int i = 0; i < batch_size * dim; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * dim * sizeof(float));
    cudaMemcpy(d_input, input, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 5; i ++) {
        benchmark_softmax(d_input, d_output, batch_size, dim);
    }
    cudaMemcpy(output, d_output, batch_size * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input;
    delete[] output;
    return 0;
}