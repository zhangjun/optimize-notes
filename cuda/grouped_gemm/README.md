# CUDA Grouped GEMM 优化

本文整理 `Grouped GEMM` 在 CUDA 上的优化思路，目标是把「一堆形状不一致的小 GEMM」从低效串行调用，推进到「高吞吐、低 launch 开销、调度稳定」的工程实现。

## 1. 什么是 Grouped GEMM

标准 GEMM：

$$
C = A \times B,\quad A\in\mathbb{R}^{M\times K},\ B\in\mathbb{R}^{K\times N},\ C\in\mathbb{R}^{M\times N}
$$

Grouped GEMM 指一次要计算多个 GEMM 任务，每个任务形状可能不同：

$$
C_g = A_g \times B_g,\quad g\in[0,G)
$$

其中每个 `g` 的 `M_g/N_g/K_g`、stride、指针都可能不一样。

常见场景：

- MoE（Mixture of Experts）中每个 expert token 数不同
- 推理服务里动态 batch 导致矩阵形状碎片化
- 图优化后出现大量 small GEMM

## 2. 性能痛点

Grouped GEMM 的核心难点不是单个 GEMM 算法，而是**任务异构 + 调度开销**：

- **Kernel launch 过多**：每个小 GEMM 单独启动 kernel，CPU 端调度开销明显
- **SM 利用率低**：小矩阵 tile 数少，GPU 很难跑满
- **负载不均衡**：大任务慢、小任务快，静态分配容易出现长尾
- **访存不连续**：各组指针离散，L2 命中和预取效果变差

因此 Grouped GEMM 优化通常分两层：

1. 提升单个 tile 计算效率（Tensor Core、流水、向量化）  
2. 提升跨 group 调度效率（分桶、persistent、work-stealing）

## 3. 与 Batched GEMM 的区别

很多人会把 Grouped GEMM 和 Batched GEMM 混用，实际差别很关键：

- **Batched GEMM**：所有 batch 形状一致（或近似一致），可用规则化 kernel
- **Grouped GEMM**：每组形状可不同，必须支持 heterogenous 参数

当 shapes 高度一致时，优先 Batched GEMM；当形状分散且动态变化明显，Grouped GEMM 更合适。

## 4. baseline：逐组调用 cuBLAS/cuBLASLt

最直接实现是循环调用：

```cpp
for (int g = 0; g < G; ++g) {
    // 每组分别 launch
    cublasGemmEx(handle, ... A[g], B[g], C[g], M[g], N[g], K[g], ...);
}
```

优点：

- 简单
- 复用成熟库

缺点：

- 小矩阵场景 launch 开销占比高
- 组间无法共享调度上下文
- 难做跨组负载均衡

这通常是功能可用基线，不是性能终点。

## 5. 优化主线

## 5.1 分桶（bucketing）减少异构度

先按形状把 group 分桶，例如按 `(M_tile, N_tile, K_tile)` 或 `(M,N,K)` 范围分组：

- 桶内使用相同 kernel 配置（tile shape、warp 数、stage 数）
- 桶间选择不同策略

收益：

- 降低分支发散
- 提高 autotune 命中率
- 更容易做 Tensor Core 专用路径

经验上，先做 bucketing 通常是 Grouped GEMM 最稳定的第一步。

## 5.2 Persistent kernel（持久化 CTA）

替代 “每组一个/多个 launch” 的方式：启动较少 CTA 常驻 SM，从全局任务队列反复取活：

```cpp
while (true) {
    int task_id = atomicAdd(&global_counter, 1);
    if (task_id >= total_tiles) break;
    // decode task -> (group, tile_m, tile_n)
    // run mma tile
}
```

优势：

- 显著降低 launch 次数
- 自动缓解大小任务混合时的长尾
- 更容易把 GPU 跑满

注意点：

- 原子计数器争用（需要合理粒度）
- 任务编码/解码成本
- CTA 内寄存器与 shared memory 占用要可控

## 5.3 Tile 级任务化（而不是 group 级）

若按 group 分配工作，大组仍会形成尾巴；更细粒度方式是把每个 group 切成 tile：

- 任务单元：`(g, tile_m, tile_n)`
- 调度粒度更细，负载更均衡
- 与 persistent + work-stealing 配合效果好

代价是任务数量增多，需要高效 task metadata 布局（SOA 通常优于 AOS）。

## 5.4 Tensor Core 路径与对齐约束

对 `fp16/bf16/int8/fp8` 场景，尽量走 Tensor Core：

- 确保 `K`、leading dimension、对齐满足硬件要求
- 采用合适的 MMA tile（如 `16x8x16` 等）
- 累加常用 FP32（视精度要求）

常见策略：

- 对齐且尺寸友好：走 TC kernel
- 不满足约束：回退 SIMT kernel

把“可走 TC 的任务”提前筛出来做独立桶，通常能明显提升整体吞吐。

## 5.5 Pipeline 与数据搬运（Ampere/Hopper）

在支持架构上：

- Ampere：`cp.async` 做 global->shared 异步搬运
- Hopper：可进一步利用 TMA + warpgroup 特性

目标是让“加载下一块数据”和“当前块 MMA”重叠，降低 memory stall。  
Grouped GEMM 中每组地址分散，这种流水优化更依赖良好的任务排序（尽量提升局部性）。

## 5.6 任务重排（reordering）提升缓存局部性

把同一 group 或相邻地址任务尽量连续执行：

- 减少 L2 抖动
- 改善 B 矩阵复用机会（特别是某些共享权重场景）
- 降低地址跳跃带来的预取失败

如果是 MoE 场景，可按 expert id + tile id 排序，通常优于原始到达顺序。

## 6. 一个实用执行框架（工程视角）

可按如下流水组织：

1. **预处理阶段（CPU/GPU）**
   - 收集每组 `M/N/K`、指针、stride
   - 过滤空组或极小组
   - 分桶并生成 tile 任务列表

2. **调度阶段（GPU）**
   - 启动 persistent kernel
   - CTA 从全局队列获取任务
   - 按任务桶选择 kernel 分支或模板实例

3. **计算阶段（GPU）**
   - shared memory 双缓冲
   - warp mma + epilogue（bias/activation 可选融合）

4. **收尾阶段**
   - 校验精度
   - 记录每桶耗时用于下一轮 autotune

这个框架在在线推理中比“每轮重新试配置”更稳定，适合长期运行服务。

## 7. 参数调优建议

关键参数：

- `CTA tile`：如 `128x128`、`128x64`、`64x128`
- `warp tile`：如 `64x64`、`64x32`
- `stage 数`：2/3/4（影响寄存器和 shared memory）
- `split-K`：K 很大时可考虑，但要平衡额外归约成本

调参建议：

- 先按桶离线 sweep，建立 `(shape bucket -> best config)` 表
- 在线仅做轻量选择，避免高频 autotune 抖动
- 对 P99 延迟敏感场景，优先稳态配置而非单点峰值吞吐

## 8. 数值与精度注意点

- `fp16/bf16` 输入时，累加建议 `fp32`（除非明确可接受精度损失）
- 组间形状差异会导致误差分布不均，校验不要只看平均误差
- 需要同时关注：
  - `max_abs_error`
  - `max_relative_error`
  - 按 group 的误差分位数（P50/P95/P99）

## 9. Benchmark 设计建议

至少覆盖三类负载：

1. **均匀小矩阵**：例如大量 `64x64x64`  
2. **长尾混合**：小矩阵 + 少量大矩阵  
3. **真实分布**：来自线上 MoE / 动态 batch 统计

建议记录：

- 总吞吐（TFLOPS）
- 平均延迟与 P95/P99
- GPU 利用率（SM active）
- kernel launch 次数
- DRAM/L2 throughput

对比基线至少包括：

- for-loop cuBLAS/cuBLASLt
- batched（可适配时）
- grouped + persistent（你的实现）

## 10. 常见坑

- **只看平均延迟**：忽略长尾会误判方案质量
- **bucket 过细**：管理开销上升，反而变慢
- **资源配置过激进**：寄存器/SMEM 过高导致 occupancy 下滑
- **任务元数据布局差**：AOS 导致解码访存低效
- **无回退路径**：遇到不对齐形状直接性能崩溃

## 11. 最小可跑实验（建议）

建议在本目录新增/补齐 `main.cu`，按下面路径递进：

1. `v0_loop_cublas`：逐组调用库函数 baseline  
2. `v1_grouped_static`：单 kernel 处理多组（静态分配）  
3. `v2_grouped_persistent`：persistent + tile 任务队列  
4. `v3_grouped_bucketed_tc`：分桶 + Tensor Core 专用路径

实验时固定随机种子，并输出：

- 正确性对比（vs FP32 reference）
- 每版本耗时
- 每版本 TFLOPS
- 不同 shape 分布下的稳定性对比

## 12. 参考方向

- CUTLASS grouped GEMM / grouped kernel 设计
- cuBLASLt grouped 或 matmul heuristic 用法
- NVIDIA CUDA C++ Best Practices Guide
- Hopper/Ampere 架构文档（cp.async, TMA, warpgroup）
