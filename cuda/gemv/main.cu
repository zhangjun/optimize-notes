#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff;
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

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
