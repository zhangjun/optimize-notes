## Softmax

https://github.com/Starmys/TritonStudyGroup/blob/main/1_CUDA_Softmax/csrc/better_softmax.cu

[online softmax推导](https://zhuanlan.zhihu.com/p/638788074)

### formula

$$Sofmax(x_i)= \frac{e^{x_i}}{\sum_j{e^{x_j}}}$$

#### naive softmax
  
直接按定义计算（不做平移）：

$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

其中分母记为：

$$
d = \sum_{j=1}^{N} e^{x_j}
$$

则：

$$
y_i = \frac{e^{x_i}}{d}
$$

#### safe softmax
  
先减去最大值再做指数，提升数值稳定性。设：

$$
m = \max_{1 \le j \le N} x_j
$$

则：

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=1}^{N} e^{x_j - m}}
$$

分母记为：

$$
d = \sum_{j=1}^{N} e^{x_j - m}
$$

所以：

$$
y_i = \frac{e^{x_i - m}}{d}
$$

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

```cpp
#define WARP_SIZE 32
#define FLT_MAX ((float)(1e10))

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


template <typename T, int REDUCE_SIZE, template<typename> class OP>
__forceinline__ __device__ T warp_reduce(T val) {
    for (int i = REDUCE_SIZE / 2; i >= 1; i >>= 1) {
        val = OP<T>()(val, __shfl_xor_sync(0xffffffff, val, i));
    }
    return val;
}
/**
 *  softmax kernel v2
 *  one warp for one row
*/
template <int threads_per_block=128>
__global__ void softmax_kernel_v2(float* input, int row, int col) {
    constexpr int warp_num = threads_per_block / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int cur_row = blockIdx.x * warp_num + warp_id;
    if (cur_row >= row) return;
    input += cur_row * col;
    int lane_id = threadIdx.x % WARP_SIZE;

    float max = -FLT_MAX;
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        max = max > input[i] ? max : input[i];
    }
    max = warp_reduce<float, WARP_SIZE, Max>(max);

    float sum = 0.f;
    for (int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    sum = warp_reduce<float, WARP_SIZE, Add>(sum);

    for (int i = lane_id; i < col; i += WARP_SIZE) {
        input[i] /= sum;
    }
}
```

### online softmax（单次遍历求 max + sum）

safe softmax 的核心是先求全局 `max`，再求 `sum(exp(x - max))`，通常需要两次扫描。
online softmax 通过维护中间状态 `(m, d)`，把这两步合并到一次遍历里：

- `m`: 当前扫描到位置的最大值
- `d`: 对应 `sum(exp(x - m))` 的归一化和

当读取新元素 `x` 时，状态更新为：

$$
m' = \max(m, x)
$$

$$
d' = d \cdot e^{m - m'} + e^{x - m'}
$$

如果是并行归约（比如 warp 内合并两个部分结果 `(m_1, d_1)`、`(m_2, d_2)`），合并公式为：

$$
m = \max(m_1, m_2), \quad d = d_1 \cdot e^{m_1 - m} + d_2 \cdot e^{m_2 - m}
$$

最后输出：

$$
softmax(x_i) = \frac{e^{x_i - m}}{d}
$$

下面给出一个和上面 `softmax_kernel_v2` 一致风格的 CUDA 版本（1 warp 处理 1 行）：

```cpp
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
```

> 说明：
> - 该实现在“统计阶段”只需要一次遍历输入（相比 safe softmax 的两次统计遍历）。
> - 为了写出最终概率，仍需再读一次 `x` 做 `exp(x - m) / d` 并写回输出，这是 softmax 输出本身无法避免的步骤。
> - 如果后续要和 top-k 融合，可复用 online `(m, d)` 状态做 fused kernel。

### 数值稳定性与误差分析

#### 为什么 online softmax 同样稳定

safe softmax 通过减去全局最大值 `m`，避免 `exp(x)` 溢出。  
online softmax 虽然是流式更新，但每一步都在“当前最大值坐标系”下维护 `d`：

- `d' = d * exp(m - m') + exp(x - m')`
- 其中 `m' = max(m, x)`，所以 `m - m' <= 0`、`x - m' <= 0`
- 因此指数项都在 `(0, 1]`，不会产生正向指数爆炸

这和 safe softmax 的稳定性本质一致，只是把“先全局 max 再全局 sum”改成了在线递推。

#### 递推不变式（invariant）

设已处理前 `t` 个元素，状态为 `(m_t, d_t)`，满足：

$$
m_t = \max_{1 \le j \le t} x_j,\quad
d_t = \sum_{j=1}^{t} e^{x_j - m_t}
$$

读入第 `t+1` 个元素 `x` 后：

$$
m_{t+1} = \max(m_t, x),\quad
d_{t+1} = d_t e^{m_t - m_{t+1}} + e^{x - m_{t+1}}
$$

把 `d_t` 展开：

$$
d_{t+1}
= \sum_{j=1}^{t} e^{x_j - m_t} e^{m_t - m_{t+1}} + e^{x - m_{t+1}}
= \sum_{j=1}^{t+1} e^{x_j - m_{t+1}}
$$

所以不变式保持成立，最终得到全局正确的 `m` 与 `d`。

#### 浮点误差特性（工程视角）

- 与 safe softmax 一样，主要误差来源是浮点加法顺序（归约树顺序不同会有 ulp 级差异）。
- `md_combine` 的结合律在实数域成立，但在浮点下只近似成立，所以不同并行划分可能有微小差别。
- `float` 通常足够；当 `col` 很大且分布极端时，可考虑：
  - 局部累计 `d` 使用 `float`，最终归约使用更稳定的树形规约；
  - 或将 `d` 升到 `double`（吞吐会下降，按需求取舍）。
- 验证时建议用 `max_abs_diff` + `mean_abs_diff` 对比 CPU reference，而不是逐元素 bitwise 相等。

### 与 FlashAttention 在线归一化的关系

FlashAttention 在 block-wise attention 里也维护在线状态，常写作 `(m_i, l_i)`：

- `m_i`: 当前 query 行在已处理 key-block 上的最大值
- `l_i`: 对应 `sum(exp(score - m_i))`

当处理新 block 得到局部 `(m_i^{new}, l_i^{new})` 时，合并公式与 online softmax 完全同构：

$$
m_i' = \max(m_i, m_i^{new}), \quad
l_i' = l_i e^{m_i - m_i'} + l_i^{new} e^{m_i^{new} - m_i'}
$$

并且输出向量（`O_i`）也按同样系数重标定后再合并：

$$
O_i' =
\frac{
l_i e^{m_i - m_i'} O_i +
l_i^{new} e^{m_i^{new} - m_i'} O_i^{new}
}{
l_i'
}
$$

可以把 online softmax 看作 FlashAttention 在线归一化在“标量 softmax”场景下的特例。

### 实现建议（从笔记到可用 kernel）

- 数据类型：
  - 输入可为 `half/bf16`，累计 `m,d` 建议 `float`。
- 并行映射：
  - `col <= 1024`：`1 warp / row` 通常最简单有效；
  - 更长序列可考虑 `1 block / row` + block reduce。
- 内存访问：
  - 尽量向量化 load/store（`float4`/`half2`），并保证对齐。
- 融合方向：
  - 若最终只需 top-k 或采样，不必写完整 softmax，可直接复用 `(m,d)` 做 fused top-k / sampling。
- 基准测试：
  - 同时比较吞吐与误差，分别对小 batch 与大 batch 测试，避免单一 case 误导结论。