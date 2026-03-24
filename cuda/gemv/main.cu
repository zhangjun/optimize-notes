#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err__) << std::endl;                   \
            std::exit(EXIT_FAILURE);                                               \
        }                                                                          \
    } while (0)

constexpr int M = 4096;
constexpr int K = 8192;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;  // 128
constexpr int TILE_K = 256;
constexpr int BATCH = 32;  // batched GEMV: Y[M, BATCH] = A[M, K] * X[K, BATCH]

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gemv_v0_naive(const float* A, const float* x, float* y, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        sum += A[row * k + j] * x[j];
    }
    y[row] = sum;
}

__global__ void gemv_v1_warp_per_row(const float* A, const float* x, float* y, int m, int k) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= m) {
        return;
    }

    float local = 0.0f;
    const int row_offset = row * k;
    for (int j = lane; j < k; j += WARP_SIZE) {
        local += A[row_offset + j] * x[j];
    }
    local = warp_reduce_sum(local);
    if (lane == 0) {
        y[row] = local;
    }
}

__global__ void gemv_v2_warp_float4(const float* A, const float* x, float* y, int m, int k) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= m) {
        return;
    }

    float local = 0.0f;
    const int row_offset = row * k;
    int k4 = (k / 4) * 4;
    for (int j = lane * 4; j < k4; j += WARP_SIZE * 4) {
        float4 av = *reinterpret_cast<const float4*>(&A[row_offset + j]);
        float4 xv = *reinterpret_cast<const float4*>(&x[j]);
        local += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
    for (int j = k4 + lane; j < k; j += WARP_SIZE) {
        local += A[row_offset + j] * x[j];
    }

    local = warp_reduce_sum(local);
    if (lane == 0) {
        y[row] = local;
    }
}

__global__ void gemv_v3_tile_x_shared(const float* A, const float* x, float* y, int m, int k) {
    __shared__ float x_tile[TILE_K];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= m) {
        return;
    }

    float local = 0.0f;
    const int row_offset = row * k;

    for (int tile_start = 0; tile_start < k; tile_start += TILE_K) {
        int valid_k = min(TILE_K, k - tile_start);

        for (int idx = threadIdx.x; idx < valid_k; idx += blockDim.x) {
            x_tile[idx] = x[tile_start + idx];
        }
        __syncthreads();

        for (int j = lane; j < valid_k; j += WARP_SIZE) {
            local += A[row_offset + tile_start + j] * x_tile[j];
        }
        __syncthreads();
    }

    local = warp_reduce_sum(local);
    if (lane == 0) {
        y[row] = local;
    }
}

void cpu_gemv_ref(const std::vector<float>& A, const std::vector<float>& x, std::vector<float>& y, int m, int k) {
    for (int i = 0; i < m; ++i) {
        float sum = 0.0f;
        const int row_offset = i * k;
        for (int j = 0; j < k; ++j) {
            sum += A[row_offset + j] * x[j];
        }
        y[i] = sum;
    }
}

void cpu_batched_gemv_ref(const std::vector<float>& A, const std::vector<float>& X, std::vector<float>& Y, int m, int k, int b) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < b; ++col) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) {
                sum += A[row * k + j] * X[j * b + col];
            }
            Y[row * b + col] = sum;
        }
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
}

std::string select_batched_kernel(int m, int k, int b, bool tensor_core_available) {
    const bool tc_shape_ok = (m % 16 == 0) && (k % 16 == 0) && (b % 16 == 0);
    if (tensor_core_available && tc_shape_ok && k >= 512 && b >= 16) {
        return "bg_tc_wmma_fp16";
    }
    if ((k % 4 == 0) && (b >= 8)) {
        return "bg_v3";
    }
    if (k >= 256) {
        return "bg_v2";
    }
    return "bg_v1";
}

__global__ void bgemv_v0_naive(const float* A, const float* X, float* Y, int m, int k, int b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= b) {
        return;
    }
    float sum = 0.0f;
    int row_offset = row * k;
    for (int j = 0; j < k; ++j) {
        sum += A[row_offset + j] * X[j * b + col];
    }
    Y[row * b + col] = sum;
}

__global__ void bgemv_v1_warp_per_output(const float* A, const float* X, float* Y, int m, int k, int b) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x;
    int col = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (row >= m || col >= b) {
        return;
    }

    float local = 0.0f;
    int row_offset = row * k;
    for (int j = lane; j < k; j += WARP_SIZE) {
        local += A[row_offset + j] * X[j * b + col];
    }
    local = warp_reduce_sum(local);
    if (lane == 0) {
        Y[row * b + col] = local;
    }
}

__global__ void bgemv_v2_tile_x_shared(const float* A, const float* X, float* Y, int m, int k, int b) {
    __shared__ float a_tile[TILE_K];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x;
    int col = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (row >= m || col >= b) {
        return;
    }

    float local = 0.0f;
    for (int tile_start = 0; tile_start < k; tile_start += TILE_K) {
        int valid_k = min(TILE_K, k - tile_start);
        for (int idx = threadIdx.x; idx < valid_k; idx += blockDim.x) {
            a_tile[idx] = A[row * k + tile_start + idx];
        }
        __syncthreads();
        for (int j = lane; j < valid_k; j += WARP_SIZE) {
            local += a_tile[j] * X[(tile_start + j) * b + col];
        }
        __syncthreads();
    }

    local = warp_reduce_sum(local);
    if (lane == 0) {
        Y[row * b + col] = local;
    }
}

__global__ void bgemv_v3_tile_x_shared_float4(const float* A, const float* X, float* Y, int m, int k, int b) {
    __shared__ float a_tile[TILE_K];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int row = blockIdx.x;
    int col = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (row >= m || col >= b) {
        return;
    }

    float local = 0.0f;
    for (int tile_start = 0; tile_start < k; tile_start += TILE_K) {
        int valid_k = min(TILE_K, k - tile_start);
        for (int idx = threadIdx.x; idx < valid_k; idx += blockDim.x) {
            a_tile[idx] = A[row * k + tile_start + idx];
        }
        __syncthreads();

        int valid_k4 = (valid_k / 4) * 4;
        for (int j = lane * 4; j < valid_k4; j += WARP_SIZE * 4) {
            float4 av = *reinterpret_cast<const float4*>(&a_tile[j]);
            float4 xv = *reinterpret_cast<const float4*>(&X[(tile_start + j) * b + col]);
            local += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
        }
        for (int j = valid_k4 + lane; j < valid_k; j += WARP_SIZE) {
            local += a_tile[j] * X[(tile_start + j) * b + col];
        }
        __syncthreads();
    }

    local = warp_reduce_sum(local);
    if (lane == 0) {
        Y[row * b + col] = local;
    }
}

__global__ void bgemv_tc_wmma(const __half* A, const __half* X, float* Y, int m, int k, int b) {
#if (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    if (tile_m * 16 >= m || tile_n * 16 >= b) {
        return;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kk = 0; kk < k; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        const __half* a_ptr = A + tile_m * 16 * k + kk;
        const __half* b_ptr = X + kk * b + tile_n * 16;
        wmma::load_matrix_sync(a_frag, a_ptr, k);
        wmma::load_matrix_sync(b_frag, b_ptr, b);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* y_ptr = Y + tile_m * 16 * b + tile_n * 16;
    wmma::store_matrix_sync(y_ptr, c_frag, b, wmma::mem_row_major);
#endif
}

float run_kernel(const std::string& version, const float* dA, const float* dx, float* dy, int m, int k, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((m + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    if (version == "v0") {
        dim3 block_v0(256);
        dim3 grid_v0((m + block_v0.x - 1) / block_v0.x);
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, stream));
        gemv_v0_naive<<<grid_v0, block_v0, 0, stream>>>(dA, dx, dy, m, k);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        return ms;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
    if (version == "v1") {
        gemv_v1_warp_per_row<<<grid, block, 0, stream>>>(dA, dx, dy, m, k);
    } else if (version == "v2") {
        gemv_v2_warp_float4<<<grid, block, 0, stream>>>(dA, dx, dy, m, k);
    } else if (version == "v3") {
        gemv_v3_tile_x_shared<<<grid, block, 0, stream>>>(dA, dx, dy, m, k);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

float run_batched_kernel(const std::string& version, const float* dA, const float* dX, float* dY, int m, int k, int b, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    if (version == "bg_v0") {
        dim3 block(16, 16);
        dim3 grid((b + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        bgemv_v0_naive<<<grid, block, 0, stream>>>(dA, dX, dY, m, k, b);
    } else if (version == "bg_v1") {
        dim3 block(BLOCK_SIZE);
        dim3 grid(m, (b + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        bgemv_v1_warp_per_output<<<grid, block, 0, stream>>>(dA, dX, dY, m, k, b);
    } else if (version == "bg_v2") {
        dim3 block(BLOCK_SIZE);
        dim3 grid(m, (b + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        bgemv_v2_tile_x_shared<<<grid, block, 0, stream>>>(dA, dX, dY, m, k, b);
    } else if (version == "bg_v3") {
        dim3 block(BLOCK_SIZE);
        dim3 grid(m, (b + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        bgemv_v3_tile_x_shared_float4<<<grid, block, 0, stream>>>(dA, dX, dY, m, k, b);
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

float run_batched_tc_kernel(const __half* dAh, const __half* dXh, float* dY, int m, int k, int b, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
    dim3 block(32, 1, 1);  // one warp per 16x16 tile
    dim3 grid((b + 15) / 16, (m + 15) / 16);
    bgemv_tc_wmma<<<grid, block, 0, stream>>>(dAh, dXh, dY, m, k, b);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::vector<float> hA(M * K);
    std::vector<float> hx(K);
    std::vector<float> hy(M, 0.0f);
    std::vector<float> hy_ref(M, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : hA) {
        v = dist(rng);
    }
    for (auto& v : hx) {
        v = dist(rng);
    }

    cpu_gemv_ref(hA, hx, hy_ref, M, K);

    float* dA = nullptr;
    float* dx = nullptr;
    float* dy = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * hA.size()));
    CHECK_CUDA(cudaMalloc(&dx, sizeof(float) * hx.size()));
    CHECK_CUDA(cudaMalloc(&dy, sizeof(float) * hy.size()));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx, hx.data(), sizeof(float) * hx.size(), cudaMemcpyHostToDevice));

    const int warmup = 20;
    const int iters = 100;
    std::vector<std::string> versions = {"v0", "v1", "v2", "v3"};

    std::cout << "M=" << M << ", K=" << K << ", BLOCK_SIZE=" << BLOCK_SIZE << ", TILE_K=" << TILE_K << std::endl;
    for (const auto& version : versions) {
        for (int i = 0; i < warmup; ++i) {
            (void)run_kernel(version, dA, dx, dy, M, K, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_kernel(version, dA, dx, dy, M, K, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }

        CHECK_CUDA(cudaMemcpy(hy.data(), dy, sizeof(float) * hy.size(), cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hy, hy_ref);
        float avg_ms = sum_ms / static_cast<float>(iters);
        double bytes = static_cast<double>(M) * K * sizeof(float) + static_cast<double>(K) * sizeof(float) + static_cast<double>(M) * sizeof(float);
        double gbps = bytes / (avg_ms * 1e6);
        std::cout << "[" << version << "] avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_abs_diff=" << diff
                  << ", effective_bandwidth(GB/s)=" << gbps << std::endl;
    }

    // Batched GEMV benchmark: Y[M, B] = A[M, K] * X[K, B]
    std::vector<float> hX_batch(K * BATCH);
    std::vector<float> hY_batch(M * BATCH, 0.0f);
    std::vector<float> hY_batch_ref(M * BATCH, 0.0f);
    for (auto& v : hX_batch) {
        v = dist(rng);
    }
    cpu_batched_gemv_ref(hA, hX_batch, hY_batch_ref, M, K, BATCH);

    float* dX_batch = nullptr;
    float* dY_batch = nullptr;
    CHECK_CUDA(cudaMalloc(&dX_batch, sizeof(float) * hX_batch.size()));
    CHECK_CUDA(cudaMalloc(&dY_batch, sizeof(float) * hY_batch.size()));
    CHECK_CUDA(cudaMemcpy(dX_batch, hX_batch.data(), sizeof(float) * hX_batch.size(), cudaMemcpyHostToDevice));

    std::vector<std::string> bg_versions = {"bg_v0", "bg_v1", "bg_v2", "bg_v3"};
    std::cout << "\nBatched GEMV benchmark (BATCH=" << BATCH << ")" << std::endl;
    for (const auto& version : bg_versions) {
        for (int i = 0; i < warmup; ++i) {
            (void)run_batched_kernel(version, dA, dX_batch, dY_batch, M, K, BATCH, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_batched_kernel(version, dA, dX_batch, dY_batch, M, K, BATCH, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }

        CHECK_CUDA(cudaMemcpy(hY_batch.data(), dY_batch, sizeof(float) * hY_batch.size(), cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hY_batch, hY_batch_ref);
        float avg_ms = sum_ms / static_cast<float>(iters);
        double bytes = static_cast<double>(M) * K * sizeof(float) + static_cast<double>(K) * BATCH * sizeof(float) +
                       static_cast<double>(M) * BATCH * sizeof(float);
        double gbps = bytes / (avg_ms * 1e6);
        std::cout << "[" << version << "] avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_abs_diff=" << diff
                  << ", effective_bandwidth(GB/s)=" << gbps << std::endl;
    }

    // Tensor Core path (FP16 input, FP32 accumulate via WMMA), requires M/K/BATCH multiples of 16.
    std::vector<__half> hA_half(M * K);
    std::vector<__half> hX_half(K * BATCH);
    for (size_t i = 0; i < hA.size(); ++i) {
        hA_half[i] = __float2half(hA[i]);
    }
    for (size_t i = 0; i < hX_batch.size(); ++i) {
        hX_half[i] = __float2half(hX_batch[i]);
    }
    __half* dA_half = nullptr;
    __half* dX_half = nullptr;
    CHECK_CUDA(cudaMalloc(&dA_half, sizeof(__half) * hA_half.size()));
    CHECK_CUDA(cudaMalloc(&dX_half, sizeof(__half) * hX_half.size()));
    CHECK_CUDA(cudaMemcpy(dA_half, hA_half.data(), sizeof(__half) * hA_half.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX_half, hX_half.data(), sizeof(__half) * hX_half.size(), cudaMemcpyHostToDevice));

    const bool tc_shape_ok = (M % 16 == 0 && K % 16 == 0 && BATCH % 16 == 0);
    if (tc_shape_ok) {
        for (int i = 0; i < warmup; ++i) {
            (void)run_batched_tc_kernel(dA_half, dX_half, dY_batch, M, K, BATCH, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_batched_tc_kernel(dA_half, dX_half, dY_batch, M, K, BATCH, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }
        CHECK_CUDA(cudaMemcpy(hY_batch.data(), dY_batch, sizeof(float) * hY_batch.size(), cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hY_batch, hY_batch_ref);
        float avg_ms = sum_ms / static_cast<float>(iters);
        double flops = 2.0 * static_cast<double>(M) * K * BATCH;
        double tflops = flops / (avg_ms * 1e9);
        std::cout << "[bg_tc_wmma_fp16] avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_abs_diff=" << diff
                  << ", effective_tflops=" << tflops << std::endl;
    } else {
        std::cout << "[bg_tc_wmma_fp16] skipped (M/K/BATCH must be multiples of 16)." << std::endl;
    }

    const std::string auto_kernel = select_batched_kernel(M, K, BATCH, true);
    std::cout << "\nAuto-picked batched kernel: " << auto_kernel << std::endl;
    if (auto_kernel == "bg_tc_wmma_fp16" && tc_shape_ok) {
        for (int i = 0; i < warmup; ++i) {
            (void)run_batched_tc_kernel(dA_half, dX_half, dY_batch, M, K, BATCH, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_batched_tc_kernel(dA_half, dX_half, dY_batch, M, K, BATCH, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }
        CHECK_CUDA(cudaMemcpy(hY_batch.data(), dY_batch, sizeof(float) * hY_batch.size(), cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hY_batch, hY_batch_ref);
        float avg_ms = sum_ms / static_cast<float>(iters);
        std::cout << "[auto] avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_abs_diff=" << diff << std::endl;
    } else {
        for (int i = 0; i < warmup; ++i) {
            (void)run_batched_kernel(auto_kernel, dA, dX_batch, dY_batch, M, K, BATCH, stream);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_batched_kernel(auto_kernel, dA, dX_batch, dY_batch, M, K, BATCH, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }
        CHECK_CUDA(cudaMemcpy(hY_batch.data(), dY_batch, sizeof(float) * hY_batch.size(), cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hY_batch, hY_batch_ref);
        float avg_ms = sum_ms / static_cast<float>(iters);
        std::cout << "[auto] avg_ms=" << avg_ms << ", min_ms=" << min_ms << ", max_abs_diff=" << diff << std::endl;
    }

    CHECK_CUDA(cudaFree(dX_batch));
    CHECK_CUDA(cudaFree(dY_batch));
    CHECK_CUDA(cudaFree(dA_half));
    CHECK_CUDA(cudaFree(dX_half));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
