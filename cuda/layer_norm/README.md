# CUDA LayerNorm 性能优化

本文整理 `LayerNorm` 在 CUDA 上的常见优化路径，目标是把「能跑」逐步推进到「高吞吐、低延迟、可扩展」。

## 1. LayerNorm 计算定义

给定输入向量 `x`（长度为 `N`），LayerNorm（忽略 batch 维）定义为：

$$
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
$$

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
$$

$$
y_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot \gamma_i + \beta_i
$$

其中：
- `gamma`、`beta` 为可学习参数（长度同 `N`）
- `epsilon` 用于数值稳定

常见等价形式（更适合实现）：

$$
\sigma^2 = E[x^2] - (E[x])^2
$$

这让我们可以只做两类归约：`sum(x)` 和 `sum(x^2)`。

## 2. 性能瓶颈直觉

LayerNorm 通常是 **memory-bound** 算子，瓶颈往往不是 FLOPs，而是全局内存带宽和访存效率：

- 需要读 `x`，写 `y`，还要读 `gamma/beta`
- 如果实现不当，`x` 可能被重复读多次
- 归约（mean/var）会引入同步与线程间通信开销

优化主线可概括为：

1. 降低访存次数（尽量一次读入，多次在寄存器/共享内存复用）  
2. 提升访存效率（向量化、对齐、连续访问）  
3. 降低归约开销（warp shuffle + 分层规约）  
4. 控制资源占用（寄存器/SMEM/occupancy 平衡）

## 3. Kernel 映射策略

最常见映射：**一行（hidden dim）对应一个 block 或多个 warp**。

- 小 hidden（如 `<= 1024`）：1 block 处理 1 row 很常见
- 中等 hidden：每线程处理多个元素（`for i += blockDim.x`）
- 大 hidden：可考虑多 block 协作同一 row（实现复杂，通常在极大维度才值得）

经验规则：

- 先让线程访问模式连续（coalesced）
- 再考虑更激进的并行拆分

## 4. 基线实现（3-pass 思路）

一个直观 baseline：

1. pass1：求 `sum(x)` 得到 mean  
2. pass2：求 `sum((x-mean)^2)` 得到 var  
3. pass3：写回 `y`

特点：

- 优点：逻辑清晰、容易验证正确性
- 缺点：访存次数多；mean/var 分两次归约开销大

这通常是「可用但不快」的起点。

## 5. 常见优化路径

### 5.1 单次加载 + 双统计

把每个线程负责的数据加载到寄存器后，同时累计：

- `sum += x`
- `sum_sq += x * x`

然后由 block 归约得到整行 `sum/sum_sq`，进而算出：

```cpp
mean = sum / N;
var = sum_sq / N - mean * mean;
inv_std = rsqrtf(var + eps);
```

收益：

- 避免单独 second pass 重新读取 `x` 来算方差
- 计算链路更短，带宽压力更小

### 5.2 Warp-level 归约替代纯 shared memory 归约

先在 warp 内用 `__shfl_down_sync` 做规约，再在 block 级做一次小规模汇总，常见收益：

- 减少 `__syncthreads()` 次数
- 降低 shared memory 读写频率
- 在中小 hidden 时延迟明显下降

### 5.3 向量化访存（`float2/float4`）

在地址对齐且 `N` 可整除向量宽度时，改成向量读写：

- `reinterpret_cast<const float4*>` 读取 `x/gamma/beta`
- 每次指令搬运更多数据，提高 L2/DRAM 利用率

注意：

- 需保证对齐（通常 16B 对齐对应 `float4`）
- 处理尾部元素（remainder）

### 5.4 融合 affine 写回

归一化后立即乘 `gamma` 加 `beta` 并写回，避免额外 kernel：

- 减少中间结果落地和再次读取
- 对推理场景很关键（减少 kernel launch + memory traffic）

### 5.5 合理选择 block size

`blockDim.x` 常见候选：`128/256/512`。不是越大越好：

- 太小：并行度不足，归约轮次多
- 太大：寄存器和 SMEM 占用上升，可能压低 occupancy

建议用 profiler（Nsight Compute）对不同 hidden 维度做 sweep。

## 6. 数值与精度注意点

### 6.1 方差计算稳定性

`var = E[x^2] - E[x]^2` 在数值上可能出现微小负值（舍入误差），常见保护：

```cpp
var = fmaxf(var, 0.f);
```

### 6.2 half/bfloat16 累加

输入是 `fp16/bf16` 时，推荐：

- 读取低精度
- 在 `fp32` 中累计 `sum/sum_sq`
- 输出再 cast 回目标精度

这通常能显著降低误差累积。

## 7. 一个实用的优化版本结构（伪代码）

```cpp
// 一个 block 处理一行
load x (vectorized, strided by tid) -> registers
local_sum += x
local_sum_sq += x * x

block_reduce(local_sum, local_sum_sq) -> sum, sum_sq
mean = sum / N
inv_std = rsqrtf(sum_sq / N - mean * mean + eps)

for each owned element:
    y = (x - mean) * inv_std
    y = y * gamma + beta
    store y
```

实现时可分三层抽象：

- `warp_reduce_sum`
- `block_reduce_sum`
- `layernorm_kernel<T, VecWidth>`

这样便于后续扩展不同数据类型与向量宽度。

## 8. 基准测试建议

至少覆盖以下维度：

- hidden size：`256, 512, 1024, 2048, 4096, 8192`
- dtype：`fp16/bf16/fp32`
- batch/token 规模：小 batch 与大 batch 各一组

观察指标：

- kernel time（us）
- effective bandwidth（GB/s）
- achieved occupancy
- DRAM throughput 与 L2 hit rate

如果优化后时间下降但带宽没升，往往是同步/指令开销优化起作用；  
若带宽显著提升，通常是访存模式和向量化收益。

## 9. 可继续深入的方向

- 针对固定 hidden 做模板特化和 unroll
- 使用 `cp.async`（Ampere+）做更细粒度流水
- 与上游算子融合（如 residual + layernorm）
- 对超长 hidden 尝试多 CTA 协同归约

## 10. 参考

- OneFlow LayerNorm 优化思路（可搜其 CUDA kernel 实现）
- Apex / xFormers / TransformerEngine 中的 LayerNorm kernel
- NVIDIA CUDA C++ Best Practices Guide

## 11. 最小可跑实验（建议）

建议在本目录新增 `main.cu`，按如下版本递进：

1. `layernorm_v0_naive_3pass`：直观 baseline  
2. `layernorm_v1_block_reduce`：单次加载统计 `sum/sum_sq`  
3. `layernorm_v2_vec4`：v1 + `float4` 向量化 + warp/block 分层规约

编译运行：

```bash
nvcc -O3 -arch=sm_80 -lineinfo -o layernorm main.cu
./layernorm
```

> 若是 `sm_90` 可改成 `-arch=sm_90`；建议先固定一张卡做版本对比，避免硬件差异干扰结论。

## 12. 关键代码骨架

下面给一个足够实用的骨架，方便直接改成可编译版本。

### 12.1 Warp / Block 归约工具

```cpp
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float smem[32];  // 最多支持 1024 threads -> 32 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    float out = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.f;
    if (wid == 0) out = warp_reduce_sum(out);
    return out;
}
```

### 12.2 v1: 单次统计 + 融合写回

```cpp
__global__ void layernorm_v1(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float local_sum = 0.f;
    float local_sq  = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_x[i];
        local_sum += v;
        local_sq += v * v;
    }

    float sum = block_reduce_sum(local_sum);
    float sq  = block_reduce_sum(local_sq);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / cols;
        float var = sq / cols - mean * mean;
        var = fmaxf(var, 0.f);
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_x[i];
        float n = (v - s_mean) * s_inv_std;
        row_y[i] = n * gamma[i] + beta[i];
    }
}
```

### 12.3 v2: 向量化（`float4`）思路

```cpp
// 仅展示核心思路：cols 可被 4 整除且地址 16B 对齐时走 float4 路径
int vec_cols = cols >> 2;  // cols / 4
const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
const float4* g4 = reinterpret_cast<const float4*>(gamma);
const float4* b4 = reinterpret_cast<const float4*>(beta);
float4* row_y4 = reinterpret_cast<float4*>(row_y);

for (int i = threadIdx.x; i < vec_cols; i += blockDim.x) {
    float4 vx = row_x4[i];
    // 统计 sum/sum_sq，并在写回阶段做 affine
    // ... 省略同模式展开
    row_y4[i] = /* normalized(vx) * g4[i] + b4[i] */;
}
```

工程里通常用模板参数 `VecSize=1/2/4` 在编译期分发，运行时根据 `cols` 和地址对齐选择路径。

## 13. Host 侧 benchmark 建议模板

### 13.1 计时方法

- 每个版本先 warmup `20` 次
- 正式跑 `100` 次，统计 avg/min
- 用 `cudaEventRecord` 做 kernel 时间测量

### 13.2 正确性校验

至少做两类校验：

- 与 CPU reference 对比（`max_abs_err`, `mean_abs_err`）
- 检查 `NaN/Inf`

CPU 参考实现建议全程 `double` 累加，减少 reference 自身误差。

### 13.3 有效带宽估算

可用简化公式估算：

$$
BW_{eff} = \frac{Bytes_{read} + Bytes_{write}}{time}
$$

对 FP32 LayerNorm（按一遍读 `x/gamma/beta` + 一遍写 `y` 粗估）：

$$
Bytes \approx rows \times cols \times 4 \times (3 + 1)
$$

即每元素约 `16 bytes`（粗估，未计缓存重用与额外访存）。  
这个值用于横向比较版本趋势，不必追求“理论精确”。

## 14. 结果记录表（可直接填）

| GPU | rows x cols | dtype | kernel | block size | time(us) | GB/s | max abs err |
|---|---:|---|---|---:|---:|---:|---:|
| A100 | 4096 x 4096 | fp32 | v0 | 256 | - | - | - |
| A100 | 4096 x 4096 | fp32 | v1 | 256 | - | - | - |
| A100 | 4096 x 4096 | fp32 | v2 | 256 | - | - | - |

建议再补一组 `fp16/bf16 + fp32 accumulate`，通常更接近推理实际。

## 15. 调参顺序建议（实用）

1. 固定 `v1`，扫 `blockDim = 128/256/512`  
2. 选最优 block 后再上 `v2(float4)`  
3. 再决定是否值得为固定 `cols` 做模板特化  
4. 最后再考虑更复杂融合（residual + layernorm）

这样能避免“同时改太多变量”导致结论混乱。

## 16. 常见问题排查

- **结果偶发 NaN**：检查 `eps`、`var` 是否 `fmaxf(var, 0.f)`、输入是否本身含 NaN  
- **v2 比 v1 慢**：大概率是对齐/尾部处理导致分支开销，或 `cols` 太小不适合向量化  
- **occupancy 很低**：检查寄存器占用（`nvcc --ptxas-options=-v`）与 block size  
- **误差偏大**：确认低精度输入是否用 FP32 累加；校验 reference 实现是否稳定  
- **不同 hidden 最优 block 不同**：正常现象，建议按 hidden 分 bucket 选配置

## 17. 下一步可以补的内容

- `main.cu` 完整可编译代码（v0/v1/v2 + benchmark + 校验）
- `fp16/bf16` 版本（含 `half2` 路径）
- `RMSNorm` 对照实验（去掉均值，只保留 RMS 统计）
