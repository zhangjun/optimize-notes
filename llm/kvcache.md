# KV Cache 显存占用和计算量

## 1. 符号定义

- `B`: batch size
- `L`: 当前已缓存的上下文长度（历史 token 数）
- `T_new`: 本次新加入的 token 数（prefill 时通常较大，decode 时常为 1）
- `N_layer`: Transformer 层数
- `N_q`: query 头数
- `N_kv`: kv 头数（MHA 时 `N_kv = N_q`，GQA/MQA 时 `N_kv < N_q`）
- `D`: 每个 head 的维度（head dim）
- `s`: 每个元素字节数（FP16/BF16 为 2，FP8 为 1）

KV cache 只缓存 K 和 V，不缓存 Q。

---

## 2. KV cache 显存占用

单层、单 token、单 batch 的 KV 元素数：

`K + V = 2 * N_kv * D`

所以总显存（字节）：

`Mem_bytes = B * L * N_layer * 2 * N_kv * D * s`

换算到 GiB：

`Mem_GiB = Mem_bytes / 1024^3`

### 关键结论

- 显存占用与 `L` 线性增长（上下文越长，KV 越大）
- 显存占用与 `N_kv` 线性增长（GQA/MQA 能显著省显存）
- 与 `N_q` 无直接线性关系（只通过 `N_kv` 进入公式）

---

## 3. 注意力计算量（核心看 decode）

下面仅看注意力主项（忽略 LN、MLP、投影常数项），用 MACs 量级描述。

### 3.1 单层 prefill（处理 `T_new` 个新 token）

prefill 会做完整的因果注意力，复杂度近似：

`O(B * N_q * D * T_new^2)`

若考虑已有缓存长度 `L`，总长度 `S = L + T_new`，则 prefill 的注意力主项约为：

`O(B * N_q * D * S * T_new)`（分块实现下常见视角）

直观上：prefill 是“二次”或“近二次”增长。

### 3.2 单层 decode（每步 1 个 token）

每步需要：

1) `Q @ K^T`，形状约 `[N_q, D] x [N_kv, L, D]`（按组映射后等价为每个 q 头与对应 kv 头做长度 `L` 的点积）  
2) `softmax(scores)`，长度 `L`  
3) `P @ V`，再与长度 `L` 做加权和

因此每层每步主复杂度：

`O(B * N_q * D * L)`

整网每步：

`O(B * N_layer * N_q * D * L)`

直观上：decode 每步随上下文长度 `L` 线性变慢。

---

## 4. 新增 KV 的带宽写入量

decode 每生成 1 个 token，需要把该 token 的 K/V 写入缓存。

每步写入字节数：

`Write_bytes_per_step = B * N_layer * 2 * N_kv * D * s`

这部分是纯线性写带宽压力，常在长序列+大 batch 下变成瓶颈之一。

---

## 5. MHA vs GQA/MQA

把 `N_kv` 从 `N_q` 降到更小（例如 GQA）会同时带来：

1) **KV 显存下降**：按 `N_kv` 比例线性下降  
2) **KV 读写带宽下降**：decode 读 K/V、写 K/V 都按 `N_kv` 下降  
3) **注意力主计算通常也更友好**（实现相关，但总体内存系统压力明显下降）

这就是大模型推理普遍偏好 GQA/MQA 的核心原因之一。

---

## 6. 快速估算示例

假设：

- `B = 1`
- `L = 32k`
- `N_layer = 32`
- `N_kv = 8`
- `D = 128`
- `s = 2`（BF16）

KV cache 显存：

`Mem_bytes = 1 * 32768 * 32 * 2 * 8 * 128 * 2 = 4,294,967,296 bytes`

即约 `4.0 GiB`。

如果改成 MHA（`N_kv = N_q = 32`，其余不变），显存约变为 `16.0 GiB`，正好 4 倍。

---

## 7. 一句话总结

- **容量角度**：KV cache 是 `O(B * L * N_layer * N_kv * D)`  
- **时延角度（decode）**：每步注意力主项是 `O(B * N_layer * N_q * D * L)`  
- 长上下文推理的关键矛盾通常是：**KV 容量 + KV 带宽 + 线性随 `L` 增长的 decode 时延**。
