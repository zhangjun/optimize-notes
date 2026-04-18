# RMSNorm

## 1. 定义

RMSNorm (Root Mean Square Layer Normalization) 的核心是只做均方根归一化，不做均值中心化。

给定输入向量 `x in R^d`，其计算为：

`rms(x) = sqrt((1 / d) * sum_i(x_i^2) + eps)`

`y_i = (x_i / rms(x)) * w_i`

其中：
- `w` 是可学习的缩放参数（与隐藏维度同长度）
- `eps` 是数值稳定项

和 LayerNorm 相比，RMSNorm 少了减均值步骤，计算更简单，通常更高效。

## 2. 前向计算流程

对每一行（token hidden state）：

1. 计算平方和 `sum(x_i^2)`
2. 求 `inv_rms = rsqrt(sum(x_i^2) / d + eps)`
3. 对每个元素做 `y_i = x_i * inv_rms * w_i`

## 3. 反向梯度（简述）

设 `g_i = dL/dy_i`，先记 `z_i = x_i * inv_rms`，则：

- `dL/dw_i = g_i * z_i`
- `dL/dx` 由两部分组成：
  - 直接项：`g_i * w_i * inv_rms`
  - 通过 `inv_rms` 链式回传的耦合项（依赖 `sum_j x_j * g_j * w_j`）

实现时通常会先做一次规约得到共享标量，再并行更新 `dx`。

## 4. CUDA 实现要点

- **并行粒度**：常见是一个 block 处理一行 hidden states，线程并行加载并累加平方和。
- **规约优化**：优先使用 warp shuffle + 分层 block reduction，减少 shared memory 压力。
- **向量化访存**：hidden size 对齐时可用 `float2/float4` 或 `half2` 提升吞吐。
- **混合精度**：输入是 fp16/bf16 时，累加建议用 fp32，避免精度损失。
- **融合机会**：可与 residual/add 等算子融合，减少 global memory 往返。

## 5. 与 LayerNorm 对比

- RMSNorm：不减均值，只按均方根缩放，算子更轻。
- LayerNorm：减均值再归一化，统计量更多，计算和访存略重。

在 LLM 中，RMSNorm 常用于替代 LayerNorm，以取得更好的训练/推理效率平衡。

## 6. 伪代码

```python
def rms_norm(x, weight, eps=1e-6):
    # x: [*, hidden_size]
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return x * rms * weight
```

## 7. 面试速记版

### 一句话

RMSNorm = 只做二范数尺度归一化（不减均值）的 LayerNorm 简化版，计算更轻、在 LLM 里很常见。

### 高频问答（30 秒）

- **Q: RMSNorm 和 LayerNorm 最大区别？**  
  A: RMSNorm 不做 `x - mean(x)`，只用 `sqrt(mean(x^2) + eps)` 归一化。

- **Q: 为什么很多 LLM 用 RMSNorm？**  
  A: 少一次均值计算与相关访存，算子更轻，工程上吞吐更友好。

- **Q: 公式怎么写？**  
  A: `y = x * rsqrt(mean(x^2) + eps) * w`。

- **Q: 数值稳定怎么做？**  
  A: `eps` 防止除零；fp16/bf16 输入时，规约（sum/mean）用 fp32 累加。

- **Q: CUDA 怎么优化？**  
  A: 一行一个 block，先并行规约 `sum(x^2)`，再广播 `inv_rms` 完成逐元素缩放；配合 warp shuffle、向量化加载和融合提升性能。

### 面试可直接背的结论

RMSNorm 在保持训练稳定性的同时，去掉了 LayerNorm 的均值中心化步骤，降低了计算和内存访问开销，因此在大模型推理中更常见，尤其适合做融合和高吞吐优化。
