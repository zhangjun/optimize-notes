## Top-K

### 纯 GPU 实现的 Top-K 方案（针对K较小场景）。该方案采用**两级规约（Two-Pass Reduction）**策略：

- 第一阶段（Per-Block Top-K）：启动多个 Block，每个 Block 负责输入数据的一部分，每个线程都拥有自己的 K 个寄存器, 然后使用了 __shared__ 内存来收集 Block 内所有线程的 Top-K，计算出该 Block 内部的 Top-K。
- 第二阶段（Global Top-K）：启动一个单独的 Block，将第一阶段所有 Block 产生的结果进行汇总，得出最终的全局 Top-K。

### Comparison with Other Top-K Kernels

|Algorithm | Best Used For | GPU Implementation Detail|
|-|-|-|
|Insertion Sort|K < 128, small batches|Single-threaded or Warp-local registers.|
|Bitonic Top-K|Medium K, power-of-2 sizes|Uses bitonic sorting networks in Shared Memory.|
|Radix Select|Large K, large N|Digit-based reduction; avoids full sort.|
|WarpSelect|Fast local selection|Uses warp-level primitives and odd-size merging networks.|

### Papers

- [RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection](https://arxiv.org/pdf/2501.14336)