// warp_reduce
template <typename T, int REDUCE_SIZE, template<typename> class OP>
__forceinline__ __device__ T warp_reduce(T val) {
    for (int i = REDUCE_SIZE / 2; i >= 1; i >>= 1) {
        val = OP<T>()(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}

struct Add {
    __device__ __forceinline__ float operator() (const float& x, const float&y) {
        return x + y;
    }
};

struct Max {
    __device__ __forceinline__ float operator() (const float& x, const float&y) {
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
    float sum_exp = shared_mem[0];

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
    offset = cur_row * dim;

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