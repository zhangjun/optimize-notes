## MNN TurboQuant TQ3/TQ4 KV Cache 量化 - 详细代码分析
### 一、算法核心：TurboQuant 原理
#### 1.1 WHT (Walsh-Hadamard Transform) 旋转变换

```cpp
// TurboQuant.hpp:996-1017
static inline void tq3_wht_forward_32(float* out, const float* in) {
    // Step 1: 符号翻转（随机化）
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] = in[i] * TQ3_SIGNS[i];  // 预计算的伪随机符号
    }
    // Step 2: 蝶形变换（5级）
    for (int step = 1; step < TQ3_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ3_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = out[j];
                float b = out[j + step];
                out[j] = a + b;
                out[j + step] = a - b;
            }
        }
    }
    // Step 3: 归一化
    const float norm = 1.0f / sqrtf((float)TQ3_BLOCK_SIZE);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        out[i] *= norm;
    }
}
```
数学意义：WHT 是一种正交变换，能量保持不变。经 WHT 旋转后的数据分布更均匀，便于 Lloyd-Max 量化器工作。

#### 1.2 Lloyd-Max 码本
```cpp
// TQ3: 3-bit (8 个码字)
static const float TQ3_CODEBOOK[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f, 
    0.2451f, 0.7560f, 1.3439f, 2.1519f
};
static const float TQ3_BOUNDARIES[7] = {
    -1.7479f, -1.0500f, -0.5005f, 0.0f, 
    0.5005f, 1.0500f, 1.7479f
};
// TQ4: 4-bit (16 个码字) - 更精细的量化
static const float TQ4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
```
关键洞察：这些码本值是针对 N(0,1) 正态分布 预计算的，经过 RMS 归一化 + WHT 变换后，数据近似服从该分布。

### 二、量化数据结构
#### 2.1 TQ3 存储格式

每个 32 元素块 → 14 bytes
┌────────────┬─────────────────────────┐
│ 2 bytes    │ 12 bytes                │
│ fp16 scale │ 压缩的 3-bit 索引         │
│ (RMS)      │ (32 × 3 = 96 bits)      │
└────────────┴─────────────────────────┘

压缩率计算：
- 原始：32 × 4 bytes (fp32) = 128 bytes
- 量化后：fp16 scale + 32 * 3 bit = 14 bytes
- 压缩比：128 / 14 ≈ 9.14x (实际 ~3.5 bits/值)

#### 2.2 3-bit 索引打包
```cpp
// TurboQuant.hpp:1056-1072
// 8 个 3-bit 索引 → 3 bytes
static inline void tq3_pack_3bit_8(uint8_t* dst, const uint8_t* idx) {
    dst[0] = (idx[0])       | (idx[1] << 3) | (idx[2] << 6);
    dst[1] = (idx[2] >> 2)  | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
    dst[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5);
}
```

### 三、量化流程 (KV Cache 写入)
#### 3.1 ProcessKey/ProcessValue 量化逻辑
```cpp
// CPUKVCacheManager.cpp:694-720
if ((mKeyQuantMode == KVQuantMode::TQ3) || (mKeyQuantMode == KVQuantMode::TQ4)) {
    int bytesPerBlock = (mKeyQuantMode == KVQuantMode::TQ3) ? TQ3_BYTES_PER_BLOCK : TQ4_BYTES_PER_BLOCK;
    int blockSize = (mKeyQuantMode == KVQuantMode::TQ3) ? TQ3_BLOCK_SIZE : TQ4_BLOCK_SIZE;
    int bytesPerSeq = (mHeadDim / blockSize) * bytesPerBlock;
    
    uint8_t* keyDst = (uint8_t*)addrOfKey(kvHead);
    for (int i = 0; i < seqLen; i++) {
        T* src = key->host<T>() + i * mKvNumHead * mHeadDim + kvHead * mHeadDim;
        uint8_t* dst = keyDst + (mPastLength + i) * bytesPerSeq;
        
        float block[TQ4_BLOCK_SIZE];
        for (int b = 0; b < mHeadDim / blockSize; b++) {
            for (int k = 0; k < blockSize; k++) {
                block[k] = (float)src[b * blockSize + k];
            }
            // 调用 TurboQuant 量化函数
            if (mKeyQuantMode == KVQuantMode::TQ3) {
                tq3_quantize_block(dst + b * bytesPerBlock, block);
            } else {
                tq4_quantize_block(dst + b * bytesPerBlock, block);
            }
        }
    }
}
```

#### 3.2 量化函数实现
```cpp
// TurboQuant.hpp:1076-1112
static inline void tq3_quantize_block(uint8_t* dst, const float* src) {
    // Step 1: 计算 RMS 缩放因子
    float sumSq = 0.0f;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        sumSq += src[i] * src[i];
    }
    float rms = sqrtf(sumSq / TQ3_BLOCK_SIZE);
    
    // 保存 scale (fp16 格式，节省 2 bytes)
    uint16_t scaleFp16 = tq3_float_to_fp16(rms);
    memcpy(dst, &scaleFp16, 2);
    
    // Step 2: 归一化
    float normalized[TQ3_BLOCK_SIZE];
    float invRms = 1.0f / rms;
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        normalized[i] = src[i] * invRms;
    }
    
    // Step 3: WHT 旋转变换
    float rotated[TQ3_BLOCK_SIZE];
    tq3_wht_forward_32(rotated, normalized);
    
    // Step 4: 查找最近码字索引
    uint8_t indices[TQ3_BLOCK_SIZE];
    for (int i = 0; i < TQ3_BLOCK_SIZE; i++) {
        indices[i] = tq3_find_nearest(rotated[i]);  // 7 个边界 → 8 选 1
    }
    
    // Step 5: 打包压缩
    for (int g = 0; g < 4; g++) {
        tq3_pack_3bit_8(dst + 2 + g * 3, indices + g * 8);
    }
}
```

### 四、Attention 计算核心：Vec_dot 融合
#### 4.1 QK 计算（关键优化）
- 传统方法：先解量化再计算

  Q (fp32) → 解量化 → K_dequant (fp32) → Q @ K_dequant

- TurboQuant 方法：在旋转域直接计算
  
  Q_rotated = WHT_forward(Q_scaled)  // 预处理一次
  
  QK = Σ_b tq3_vec_dot_block(Q_rotated[b], K_tq3[b])  // 无需解量化！

```cpp
// TurboQuant.hpp:1176-1192
static inline float tq3_vec_dot_block(const float* q_rotated, const uint8_t* src) {
    // 读取 scale
    uint16_t scaleFp16;
    memcpy(&scaleFp16, src, 2);
    float scale = tq3_fp16_to_float(scaleFp16);
    
    // 直接在旋转域计算点积
    float dot = 0.0f;
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        tq3_unpack_3bit_8(indices, src + 2 + g * 3);
        for (int k = 0; k < 8; k++) {
            dot += q_rotated[g * 8 + k] * TQ3_CODEBOOK[indices[k]];
        }
    }
    return dot * scale;
}
```

数学推导：
Q · K_dequant 
= Q · (scale * WHT_inverse(codebook[idx]))
= scale * (WHT_forward(Q)) · codebook[idx]
= scale * Q_rotated · codebook[idx]

#### 4.2 CPUAttention.cpp 中的 QK 计算
```cpp
// CPUAttention.cpp:699-770 (TQ3 路径)
// 预处理：首次 KV block 时旋转 Q
if (i == 0) {
    float qScale = 1.0f / sqrtf((float)mHeadDim);
    for (int q = 0; q < seqLen; q++) {
        for (int b = 0; b < numBlocks; b++) {
            float scaled[TQ3_BLOCK_SIZE];
            // ... 加载并缩放 Q
            tq3_wht_forward_32(qRotated + q * mHeadDim + b * TQ3_BLOCK_SIZE, scaled);
        }
    }
}
// 计算 QK：直接在旋转域进行
for (int s = 0; s < subKvSeqLen; s++) {
    auto kPtr = (uint8_t*)keyAddr + seqIdx * tq3BytesPerSeq;
    for (int q = 0; q < seqLen; q++) {
        float score = 0.0f;
        auto qr = qRotated + q * mHeadDim;
        for (int b = 0; b < numBlocks; b++) {
            score += tq3_vec_dot_block(qr + b * TQ3_BLOCK_SIZE, 
                                        kPtr + b * TQ3_BYTES_PER_BLOCK);
        }
        // 写入 qkPacked
    }
}
```

#### 4.3 Value 计算（加权累积融合）
```cpp
// CPUAttention.cpp:854-920
// 核心思想：在旋转域累积，最后做一次逆变换
for (int q = rowStart; q < seqLen; q++) {
    // 提取 softmax 权重
    float* weights = weightsPtr;
    for (int s = 0; s < subKvSeqLen; s++) {
        weights[s] = ((float*)qkSoftmax)[packIdx];
    }
    
    // 每个维度块：在旋转域加权累积
    for (int b = 0; b < numBlocks; b++) {
        memset(vAccRotated, 0, TQ3_BLOCK_SIZE * sizeof(float));
        for (int s = 0; s < subKvSeqLen; s++) {
            const uint8_t* block = valueAddr + seqIdx * tq3BytesPerSeq + b * TQ3_BYTES_PER_BLOCK;
            uint16_t scaleFp16;
            memcpy(&scaleFp16, block, 2);
            float w = weights[s] * tq3_fp16_to_float(scaleFp16);
            // 旋转域加权累积
            tq3_weighted_acc_block(vAccRotated, w, block + 2);
        }
        // WHT 逆变换一次得到最终值
        float reconstructed[TQ3_BLOCK_SIZE];
        tq3_wht_inverse_32(reconstructed, vAccRotated);
        // 写入输出
    }
}
```