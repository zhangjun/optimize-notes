# CUDA GEMV 优化

本文整理 `GEMV`（General Matrix Vector Multiply）在 CUDA 上的常见优化路径，目标是从「可运行」逐步推进到「高带宽利用率 + 稳定低延迟」。

## 1. GEMV 定义与访存特性

以最常见的 `y = A * x` 为例：

- `A` 形状：`M x K`
- `x` 形状：`K`
- `y` 形状：`M`

数学形式：

$$
y_i = \sum_{j=0}^{K-1} A_{i,j} \cdot x_j,\quad i\in[0, M)
$$

若用 FP32，理想化访存量近似为：

- 读取 `A`：`M*K*4` bytes
- 读取 `x`：`K*4` bytes（实际往往会被重复读取，取决于缓存/复用策略）
- 写回 `y`：`M*4` bytes

GEMV 的一个关键直觉：**通常是 memory-bound**。  
因为每个 `A[i,j]` 只参与一次乘加，算术强度（FLOPs/Byte）不高，优化核心是让带宽更“饱和”。

## 2. baseline 映射（每行一个线程）

最直观的写法是每个线程算一行 `y_i`：

```cpp
__global__ void gemv_naive(const float* A, const float* x, float* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float sum = 0.f;
    for (int j = 0; j < K; ++j) {
        sum += A[row * K + j] * x[j];
    }
    y[row] = sum;
}
```

问题：

- `x[j]` 被大量线程重复读取，依赖缓存命中，稳定性一般
- 单线程做整行累加，线程级并行度不足
- 当 `M` 较小、`K` 较大时，SM 利用率容易偏低

## 3. 优化主线

GEMV 常见优化可以按下面顺序推进。

### 3.1 访问模式先做对（coalesced）

核心目标：

- 读取 `A` 时尽量让同一 warp 线程访问连续地址
- 避免不规则 stride 导致的内存事务放大

对于 row-major `A`，常见做法是：**一个 warp/一个 block 协作计算一行或多行**，让线程在 `K` 维上并行切分。

### 3.2 线程协作同一行（warp-level reduction）

让一个 warp 的 32 个线程共同计算一行：

- lane `t` 负责 `j = t, t+32, t+64, ...`
- 每个线程得到部分和后，用 `__shfl_down_sync` 做 warp 内规约

这样可显著提升并行度，尤其在 `K` 大时通常优于“每行一个线程”。

示例片段：

```cpp
float local = 0.f;
for (int j = lane; j < K; j += 32) {
    local += A[row * K + j] * x[j];
}
for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, offset);
}
if (lane == 0) y[row] = local;
```

### 3.3 向量化读取（`float4`）

在地址对齐且 `K` 可整除向量宽度时，可改成 `float4` 读取：

- 减少 load 指令条数
- 提高 L2/DRAM 吞吐利用率

注意：

- 确保 16-byte 对齐（`float4`）
- 处理尾部 remainder（`K % 4 != 0`）

### 3.4 `x` 复用：cache / shared memory / 常量内存

`x` 会被所有行复用，是优化重点。

- 小 `K`：可尝试放 `__constant__`（广播友好）
- 中大 `K`：分块加载 `x` 到 shared memory，再做 tile 计算
- 现代 GPU 下，纯依赖 L2 也可能不错，需以 benchmark 为准

经验：`x` 很大时，shared memory 分块常常带来更稳定收益。

### 3.5 分块 K 维（tile-k）

按 `TILE_K` 分段处理：

1. 一个 block 先把 `x[k0:k0+TILE_K)` 搬到 shared memory
2. 再读取对应的 `A[:, k0:k0+TILE_K)` 做局部累加
3. 滚动下一段

收益：

- 降低 `x` 的重复全局访存
- 提升数据局部性与带宽效率

### 3.6 资源平衡（occupancy / registers / smem）

GEMV 不一定 occupancy 越高越好，常要平衡：

- block size
- 每线程寄存器占用
- shared memory 大小

如果寄存器过高导致 active warps 降太多，吞吐会掉；建议用 Nsight Compute 看 `sm__warps_active`、`dram__throughput`、`l2_tex__throughput`。

## 4. 一个常用 kernel 映射策略

场景：`A` row-major，`y = A * x`，`K` 较大。

- 一个 warp 负责一行（warp-per-row）
- 一个 block 放多个 warp（如 4 或 8 个）
- `blockDim.x = 128/256`

优点：

- 实现简单
- 易于与向量化叠加
- 对大多数“长 K”场景有稳定表现

局限：

- `M` 很小时并行度可能不足
- `K` 很小时 lane 利用率低，可能被其它映射反超

## 5. Batched GEMV 优化

当需要同时计算多个向量时：

$$
Y = A \cdot X,\quad A\in\mathbb{R}^{M\times K},\ X\in\mathbb{R}^{K\times B},\ Y\in\mathbb{R}^{M\times B}
$$

这可以理解为 **batched GEMV**（也可视为小 `N` 的 GEMM）。

核心收益：

- 提升并行度：把原本“单向量”扩展到 `B` 个向量并行
- 更好隐藏访存延迟
- 更容易走到 Tensor Core 友好的计算形态

本文代码中的 batched 版本（`main.cu`）包含：

- `bg_v0`: naive（每线程算一个 `Y[row, col]`）
- `bg_v1`: warp-per-output（每个 warp 协作一个输出）
- `bg_v2`: `A` 分块到 shared memory，减少重复读取
- `bg_v3`: `bg_v2` + `float4` 向量化读取（更高吞吐）

## 6. Tensor Core 优化（WMMA）

对 FP16/BF16 输入、FP32 累加场景，常见路径是把 batched GEMV 重写成小 GEMM，再用 Tensor Core：

- 数据类型：`__half` 输入
- 累加：FP32 accumulator
- tile：`16x16x16`（WMMA 基础 tile）

代码中给了 `bg_tc_wmma_fp16` 示例（`wmma::mma_sync`），适用条件：

- `M`、`K`、`BATCH` 最好是 16 的倍数
- GPU 架构支持 Tensor Core（Volta+）

实践建议：

- 用 FP32 reference 做误差对比（`max_abs_diff`）
- 观察吞吐指标（TFLOPS）而不是只看时间
- 注意 half 精度误差会高于纯 FP32 路径

## 7. 自动选择策略（工程实用）

`main.cu` 增加了一个简化版自动选择器（`select_batched_kernel`），用于在 batched 路径里根据形状与硬件条件快速决策：

- 若 Tensor Core 可用，且 `M/K/BATCH` 都是 16 的倍数，同时 `K`、`BATCH` 足够大：优先 `bg_tc_wmma_fp16`
- 否则若 `K % 4 == 0` 且 `BATCH >= 8`：选 `bg_v3`（tile + float4）
- 否则 `K >= 256`：选 `bg_v2`
- 其它小规模场景：退回 `bg_v1`

这个规则是经验启发式，不同 GPU（如 Ada/Hopper）阈值可能不同，建议用 Nsight 或实测结果再微调阈值。

## 8. 最小可跑实验建议

可对比以下版本（单向量 GEMV）：

1. `v0`: 每行一个线程（naive）
2. `v1`: warp-per-row + shuffle reduce
3. `v2`: v1 + `float4` 向量化
4. `v3`: v2 + `x` 分块共享内存

并额外对比 batched/Tensor Core：

1. `bg_v0`: batched naive
2. `bg_v1`: batched warp-per-output
3. `bg_v2`: batched + tile shared memory
4. `bg_v3`: batched + tile shared memory + float4
5. `bg_tc_wmma_fp16`: batched Tensor Core

建议固定：

- warmup：20 次
- benchmark：100 次，记录 avg/min
- 与 CPU reference 校验最大误差（如 `1e-4`）

编译示例：

```bash
nvcc -O3 -arch=sm_80 -lineinfo -o gemv main.cu
./gemv
```

> 若要跑 WMMA/Tensor Core 路径，`-arch` 需至少支持 Tensor Core（如 `sm_70+`，推荐 `sm_80`）。

## 9. 常见坑

- **对齐问题**：`float4` 读取地址未对齐会退化甚至非法访问
- **边界处理**：`M`、`K` 非 block/warp 整倍数时要 guard
- **精度误差**：并行规约顺序变化会引入微小浮点差异
- **过度用 shared memory**：smem 太大可能压低 occupancy
- **只看 kernel 时间不看带宽**：GEMV 优化应结合有效带宽（GB/s）评估

## 10. 进一步方向

- 混合精度（FP16/BF16 输入 + FP32 accumulate）
- batched GEMV 的自适应调度（按 `M/K/B` 动态选 kernel）
- Tensor Core 路径的双缓冲（cp.async + pipeline）
- 融合算子（GEMV + bias/activation）减少中间写回

## 11. 参考

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
