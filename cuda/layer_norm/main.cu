#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e = (cmd);                                                 \
        if (e != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(e)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct EvalResult {
    float max_abs_err = 0.0f;
    float mean_abs_err = 0.0f;
    bool has_nan_or_inf = false;
};

struct PerfResult {
    float avg_ms = 0.0f;
    float min_ms = 0.0f;
};

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float block_reduce_sum(float v) {
    __shared__ float smem[32];  // up to 1024 threads
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        smem[warp_id] = v;
    }
    __syncthreads();

    float out = (threadIdx.x < num_warps) ? smem[lane] : 0.0f;
    if (warp_id == 0) {
        out = warp_reduce_sum(out);
    }
    return out;
}

__global__ void layernorm_v0_naive_3pass(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y, int rows,
                                         int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += row_x[i];
    }
    float sum = block_reduce_sum(local_sum);

    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / static_cast<float>(cols);
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = row_x[i] - s_mean;
        local_var_sum += d * d;
    }
    float var_sum = block_reduce_sum(local_var_sum);

    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float var = var_sum / static_cast<float>(cols);
        var = fmaxf(var, 0.0f);
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float n = (row_x[i] - s_mean) * s_inv_std;
        row_y[i] = n * gamma[i] + beta[i];
    }
}

__global__ void layernorm_v1_block_reduce(const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          float* __restrict__ y, int rows,
                                          int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_x[i];
        local_sum += v;
        local_sum_sq += v * v;
    }

    float sum = block_reduce_sum(local_sum);
    float sum_sq = block_reduce_sum(local_sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / static_cast<float>(cols);
        float var = sum_sq / static_cast<float>(cols) - mean * mean;
        var = fmaxf(var, 0.0f);
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float n = (row_x[i] - s_mean) * s_inv_std;
        row_y[i] = n * gamma[i] + beta[i];
    }
}

__global__ void layernorm_v2_vec4(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ y, int rows, int cols,
                                  float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;
    int vec_cols = cols >> 2;

    const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* row_y4 = reinterpret_cast<float4*>(row_y);

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += blockDim.x) {
        float4 v = row_x4[i];
        local_sum += (v.x + v.y + v.z + v.w);
        local_sum_sq += (v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    }

    float sum = block_reduce_sum(local_sum);
    float sum_sq = block_reduce_sum(local_sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / static_cast<float>(cols);
        float var = sum_sq / static_cast<float>(cols) - mean * mean;
        var = fmaxf(var, 0.0f);
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < vec_cols; i += blockDim.x) {
        float4 v = row_x4[i];
        float4 g = gamma4[i];
        float4 b = beta4[i];

        float4 out;
        out.x = (v.x - s_mean) * s_inv_std * g.x + b.x;
        out.y = (v.y - s_mean) * s_inv_std * g.y + b.y;
        out.z = (v.z - s_mean) * s_inv_std * g.z + b.z;
        out.w = (v.w - s_mean) * s_inv_std * g.w + b.w;
        row_y4[i] = out;
    }
}

void layernorm_cpu_ref(const std::vector<float>& x, const std::vector<float>& gamma,
                       const std::vector<float>& beta, std::vector<float>& y, int rows,
                       int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row_x = x.data() + static_cast<size_t>(r) * cols;
        float* row_y = y.data() + static_cast<size_t>(r) * cols;

        double sum = 0.0;
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            double v = static_cast<double>(row_x[c]);
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / static_cast<double>(cols);
        double var = sum_sq / static_cast<double>(cols) - mean * mean;
        var = std::max(var, 0.0);
        double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

        for (int c = 0; c < cols; ++c) {
            double n = (static_cast<double>(row_x[c]) - mean) * inv_std;
            row_y[c] = static_cast<float>(n * gamma[c] + beta[c]);
        }
    }
}

EvalResult evaluate(const std::vector<float>& got, const std::vector<float>& ref) {
    EvalResult out;
    double sum_abs_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float a = got[i];
        float b = ref[i];
        if (!std::isfinite(a)) {
            out.has_nan_or_inf = true;
        }
        float abs_err = std::fabs(a - b);
        out.max_abs_err = std::max(out.max_abs_err, abs_err);
        sum_abs_err += abs_err;
    }
    out.mean_abs_err =
        static_cast<float>(sum_abs_err / static_cast<double>(got.size()));
    return out;
}

PerfResult benchmark_kernel(const std::string& name, const float* d_x,
                           const float* d_gamma, const float* d_beta, float* d_y,
                           int rows, int cols, float eps, int block_size,
                           int warmup, int iters, cudaStream_t stream) {
    dim3 grid(rows);
    dim3 block(block_size);

    auto launch = [&]() {
        if (name == "v0") {
            layernorm_v0_naive_3pass<<<grid, block, 0, stream>>>(
                d_x, d_gamma, d_beta, d_y, rows, cols, eps);
        } else if (name == "v1") {
            layernorm_v1_block_reduce<<<grid, block, 0, stream>>>(
                d_x, d_gamma, d_beta, d_y, rows, cols, eps);
        } else if (name == "v2") {
            layernorm_v2_vec4<<<grid, block, 0, stream>>>(
                d_x, d_gamma, d_beta, d_y, rows, cols, eps);
        }
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    float min_ms = std::numeric_limits<float>::max();
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return PerfResult{total_ms / static_cast<float>(iters), min_ms};
}

double estimated_bandwidth_gbps(int64_t rows, int64_t cols, int bytes_per_elem,
                                float ms) {
    double bytes = static_cast<double>(rows) * static_cast<double>(cols) *
                   static_cast<double>(bytes_per_elem);
    return bytes / (static_cast<double>(ms) * 1.0e-3) / 1.0e9;
}

int parse_int_arg(char* s, int fallback) {
    if (s == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(v);
}

int main(int argc, char** argv) {
    int rows = (argc > 1) ? parse_int_arg(argv[1], 4096) : 4096;
    int cols = (argc > 2) ? parse_int_arg(argv[2], 4096) : 4096;
    int block_size = (argc > 3) ? parse_int_arg(argv[3], 256) : 256;
    int warmup = (argc > 4) ? parse_int_arg(argv[4], 20) : 20;
    int iters = (argc > 5) ? parse_int_arg(argv[5], 100) : 100;
    float eps = 1e-5f;

    if (block_size > 1024) {
        std::cerr << "block_size must be <= 1024" << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t numel = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<float> h_x(numel);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    std::vector<float> h_y_ref(numel);
    std::vector<float> h_y_out(numel);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_g(0.5f, 1.5f);
    std::uniform_real_distribution<float> dist_b(-0.2f, 0.2f);

    for (size_t i = 0; i < numel; ++i) {
        h_x[i] = dist_x(rng);
    }
    for (int i = 0; i < cols; ++i) {
        h_gamma[i] = dist_g(rng);
        h_beta[i] = dist_b(rng);
    }

    layernorm_cpu_ref(h_x, h_gamma, h_beta, h_y_ref, rows, cols, eps);

    float* d_x = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, static_cast<size_t>(cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, static_cast<size_t>(cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, numel * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), numel * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), static_cast<size_t>(cols) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), static_cast<size_t>(cols) * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::cout << "LayerNorm benchmark" << std::endl;
    std::cout << "rows=" << rows << ", cols=" << cols
              << ", block_size=" << block_size
              << ", warmup=" << warmup << ", iters=" << iters << std::endl;
    std::cout << "eps=" << eps << std::endl;

    std::vector<std::string> versions = {"v0", "v1"};
    bool can_run_v2 = (cols % 4 == 0);
    if (can_run_v2) {
        versions.push_back("v2");
    } else {
        std::cout << "[v2] skipped (cols must be divisible by 4 for float4 path)"
                  << std::endl;
    }

    for (const auto& v : versions) {
        PerfResult perf = benchmark_kernel(v, d_x, d_gamma, d_beta, d_y, rows, cols,
                                           eps, block_size, warmup, iters, stream);

        CUDA_CHECK(cudaMemcpy(h_y_out.data(), d_y, numel * sizeof(float),
                              cudaMemcpyDeviceToHost));
        EvalResult eval = evaluate(h_y_out, h_y_ref);

        int bytes_per_elem = (v == "v0") ? 24 : 16;
        double bw_avg = estimated_bandwidth_gbps(rows, cols, bytes_per_elem, perf.avg_ms);
        double bw_min = estimated_bandwidth_gbps(rows, cols, bytes_per_elem, perf.min_ms);

        std::cout << "[" << v << "] "
                  << "avg_ms=" << perf.avg_ms
                  << ", min_ms=" << perf.min_ms
                  << ", est_BW_avg(GB/s)=" << bw_avg
                  << ", est_BW_min(GB/s)=" << bw_min
                  << ", max_abs_err=" << eval.max_abs_err
                  << ", mean_abs_err=" << eval.mean_abs_err
                  << ", has_nan_or_inf=" << (eval.has_nan_or_inf ? "true" : "false")
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
