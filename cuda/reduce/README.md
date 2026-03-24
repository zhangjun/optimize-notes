# CUDA Reduce 算子优化

这个目录用一个简单的 block-level sum reduction 演示 CUDA Reduce 的基础优化路径，`main.cu` 里有 6 个版本：

- `reduce_v0`: 交错索引（interleaved）+ `%` 取模判断
- `reduce_v1`: 去掉 `%`，改成 `index = 2 * s * tid`
- `reduce_v2`: 改为顺序地址（sequential addressing）+ 二分规约
- `reduce_v3`: warp-level shuffle 规约，减少 block 内同步开销
- `reduce_v4`: 每线程处理 2 个元素 + warp-level shuffle
- `reduce_v5`: `float4` 向量化加载 + warp-level shuffle

## 编译与运行

```bash
nvcc -o main main.cu -arch=sm_80
./main
```

程序会自动：

- 计算 CPU reference
- 对 `v0/v1/v2/v3/v4/v5` 分别做 warmup + 多次 benchmark（打印 avg/min）
- 做 GPU 结果与 CPU 结果的一致性校验

## 代码结构速览

- 输入规模: `N = 32 * 1024 * 1024`
- block 大小: `BLOCK_SIZE = 256`
- 每个 block 输出 1 个部分和到 `g_odata`
- 使用多阶段规约（multi-pass），把每个 block 的部分和持续规约到最终单值

## 版本对比与优化点

### v0: baseline（可读性好，但效率一般）

核心逻辑：

- 每轮步长 `s` 从 1, 2, 4 ... 增长
- 只有 `tid % (2*s) == 0` 的线程参与累加

问题：

- `%` 在 GPU 上代价较高
- 活跃线程分布稀疏，warp 内分歧明显
- shared memory 访问模式不够友好

### v1: 去 `%` 优化

把判断条件改成：

```cpp
int index = 2 * s * tid;
if (index < blockDim.x) { ... }
```

收益：

- 避免取模运算
- 指令层面更轻量

局限：

- 线程活跃模式仍是“交错”的，warp divergence 仍然存在

### v2: sequential addressing（更常见的高效写法）

循环改为从 `blockDim.x / 2` 递减到 1：

```cpp
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

收益：

- 访问模式连续，shared memory bank conflict 风险更低
- 每一轮活跃线程集中在前半段，warp 执行效率更高
- 通常是前三版中最快的基础实现

### v3: warp shuffle（进一步减少同步）

思路：

- 先在每个 warp 内用 `__shfl_down_sync` 做规约
- 每个 warp 的 lane0 把部分和写入 shared memory
- 再由 warp0 对这些部分和做一次 warp 规约

收益：

- 避免了 `v2` 中从 256 到 1 每轮都 `__syncthreads()` 的开销
- block 级同步仅保留 1 次（写入 warp partial 之后）
- 通常在中大 block size 下比 `v2` 更快

### v4: 2 elements/thread + warp shuffle

思路：

- 每个线程从 global memory 读取 2 个元素并先在寄存器内求和
- 第一阶段 block 数从 `N / BLOCK_SIZE` 下降为 `N / (2 * BLOCK_SIZE)`（近似）
- block 内继续使用 `v3` 的 warp shuffle 规约

收益：

- 降低第一阶段 block 数量，减少写回的 partial sums
- 降低后续最终规约阶段的输入规模
- 通常进一步提升吞吐，尤其在 memory bound 场景

### v5: float4 vectorized load + warp shuffle

思路：

- 每个线程优先读取一个 `float4`（4 个连续元素）并在寄存器内累加
- 尾部不足 4 个元素时回退到标量边界处理
- block 内仍使用 warp shuffle 规约

收益：

- 减少全局内存 load 指令条数
- 提升访存吞吐利用率（对齐良好、顺序访问时收益更稳定）
- 在大规模输入下通常比 `v4` 更快

## 还能继续优化什么

在 `v5` 基础上，常见还能继续做：

- 每线程加载 2 个或更多元素（减少 grid 规模和同步开销）
- 最后一个 warp 使用 `warp shuffle`，减少 `__syncthreads()`
- 模板化展开（unroll）最后几轮规约
- 循环展开（unroll）+ 指令级并行（ILP）进一步压榨吞吐
- 使用 CUB / cooperative groups 做工程化对比
- 多阶段 kernel 或单 kernel + 原子操作完成全局最终规约

## 参考

- https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/reduce