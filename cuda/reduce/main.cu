#include "cuda_runtime.h"
#include "timer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <time.h>

// https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/reduce

#define CEIL_DIV(x, y) (x + y - 1) / y
__global__ void empty() {}


#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 32*1024*1024
#define BLOCK_SIZE 256

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_stage_generic(float* g_idata, float* g_odata, int n) {
    __shared__ float warp_sums[32];
    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float val = 0.0f;
    if (start < static_cast<unsigned int>(n)) {
        val += g_idata[start];
    }
    if (start + blockDim.x < static_cast<unsigned int>(n)) {
        val += g_idata[start + blockDim.x];
    }

    val = warp_reduce_sum(val);
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = val;
    }
    __syncthreads();

    if (tid < 32) {
        int warp_count = blockDim.x / 32;
        val = (tid < warp_count) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
    }
}

__global__ void reduce_v0(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v1(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // if (tid % (2*s) == 0) {
        //     sdata[tid] += sdata[tid + s];
        // }
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v2(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v3(float *g_idata,float *g_odata){
    __shared__ float warp_sums[32];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = g_idata[i];

    // First stage: intra-warp reduction via shuffle.
    val = warp_reduce_sum(val);

    // One lane writes one partial sum per warp.
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = val;
    }
    __syncthreads();

    // Final stage: warp 0 reduces all warp partial sums.
    if (tid < 32) {
        int warp_count = blockDim.x / 32;
        val = (tid < warp_count) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
    }
}

__global__ void reduce_v4(float *g_idata,float *g_odata, int n){
    __shared__ float warp_sums[32];
    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Grid-stride (2 elements per thread) style load for better bandwidth utilization.
    float val = 0.0f;
    if (start < static_cast<unsigned int>(n)) {
        val += g_idata[start];
    }
    if (start + blockDim.x < static_cast<unsigned int>(n)) {
        val += g_idata[start + blockDim.x];
    }

    val = warp_reduce_sum(val);
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = val;
    }
    __syncthreads();

    if (tid < 32) {
        int warp_count = blockDim.x / 32;
        val = (tid < warp_count) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
    }
}

__global__ void reduce_v5(float *g_idata, float *g_odata, int n) {
    __shared__ float warp_sums[32];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    // Vectorized path: one float4 per thread when fully in-bounds.
    float val = 0.0f;
    if (base + 3 < static_cast<unsigned int>(n)) {
        float4 v = reinterpret_cast<float4*>(g_idata)[base >> 2];
        val = v.x + v.y + v.z + v.w;
    } else {
        if (base < static_cast<unsigned int>(n)) val += g_idata[base];
        if (base + 1 < static_cast<unsigned int>(n)) val += g_idata[base + 1];
        if (base + 2 < static_cast<unsigned int>(n)) val += g_idata[base + 2];
        if (base + 3 < static_cast<unsigned int>(n)) val += g_idata[base + 3];
    }

    val = warp_reduce_sum(val);
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = val;
    }
    __syncthreads();

    if (tid < 32) {
        int warp_count = blockDim.x / 32;
        val = (tid < warp_count) ? warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            g_odata[blockIdx.x] = val;
        }
    }
}

bool compare_diff(const float *A, const float *B, int n) {
  for (int i = 0; i < n; ++i) {
    if (std::fabs(A[i] - B[i]) > 1e-2f) {
      return true;
    }
  }
  return false;
}

float run_reduce_kernel(const std::string& name, float* input_device,
                        float* partial_output_device, float* scratch_output_device,
                        float* final_output_device,
                        int block_num, cudaStream_t stream) {
    int stage1_blocks = block_num;
    if (name == "v4") {
        stage1_blocks = CEIL_DIV(N, BLOCK_SIZE * 2);
    } else if (name == "v5") {
        stage1_blocks = CEIL_DIV(N, BLOCK_SIZE * 4);
    }

    dim3 grid(stage1_blocks, 1);
    dim3 block(BLOCK_SIZE, 1);
    GPUTimer gpu_timer;

    gpu_timer.Start(stream);
    if (name == "v0") {
        reduce_v0<<<grid, block, 0, stream>>>(input_device, partial_output_device);
    } else if (name == "v1") {
        reduce_v1<<<grid, block, 0, stream>>>(input_device, partial_output_device);
    } else if (name == "v2") {
        reduce_v2<<<grid, block, 0, stream>>>(input_device, partial_output_device);
    } else if (name == "v3") {
        reduce_v3<<<grid, block, 0, stream>>>(input_device, partial_output_device);
    } else if (name == "v4") {
        reduce_v4<<<grid, block, 0, stream>>>(input_device, partial_output_device, N);
    } else if (name == "v5") {
        reduce_v5<<<grid, block, 0, stream>>>(input_device, partial_output_device, N);
    }

    // Multi-stage reduction until one scalar remains.
    float* in_ptr = partial_output_device;
    float* out_ptr = scratch_output_device;
    int cur_n = stage1_blocks;
    while (cur_n > 1) {
        int next_blocks = CEIL_DIV(cur_n, BLOCK_SIZE * 2);
        dim3 next_grid(next_blocks, 1);
        reduce_stage_generic<<<next_grid, block, 0, stream>>>(in_ptr, out_ptr, cur_n);
        cur_n = next_blocks;
        float* tmp = in_ptr;
        in_ptr = out_ptr;
        out_ptr = tmp;
    }
    cudaMemcpyAsync(final_output_device, in_ptr, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    gpu_timer.Stop(stream);
    return gpu_timer.ElapsedTime();
}

int main() {
    srand(time(0));
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0f;

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *partial_output_device;
    cudaMalloc((void **)&partial_output_device, block_num * sizeof(float));
    float *scratch_output_device;
    cudaMalloc((void **)&scratch_output_device, block_num * sizeof(float));
    float *final_output_device;
    cudaMalloc((void **)&final_output_device, sizeof(float));

    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    float cpu_ref = 0.0f;
    for (int i = 0; i < N; ++i) cpu_ref += input_host[i];

    std::vector<std::string> versions = {"v0", "v1", "v2", "v3", "v4", "v5"};
    int warmup = 3;
    int iters = 20;
    float gpu_out = 0.0f;

    std::cout << "N=" << N << ", BLOCK_SIZE=" << BLOCK_SIZE
              << ", block_num=" << block_num << std::endl;
    std::cout << "CPU reference: " << cpu_ref << std::endl;

    for (const auto& version : versions) {
        for (int i = 0; i < warmup; ++i) {
            (void)run_reduce_kernel(version, input_device, partial_output_device,
                                    scratch_output_device, final_output_device,
                                    block_num, stream);
        }
        cudaStreamSynchronize(stream);

        float sum_ms = 0.0f;
        float min_ms = std::numeric_limits<float>::max();
        for (int i = 0; i < iters; ++i) {
            float ms = run_reduce_kernel(version, input_device, partial_output_device,
                                         scratch_output_device, final_output_device,
                                         block_num, stream);
            sum_ms += ms;
            min_ms = std::min(min_ms, ms);
        }
        cudaStreamSynchronize(stream);
        cudaMemcpy(&gpu_out, final_output_device, sizeof(float), cudaMemcpyDeviceToHost);

        bool mismatch = compare_diff(&gpu_out, &cpu_ref, 1);
        std::cout << "[" << version << "] "
                  << "avg_ms=" << (sum_ms / iters)
                  << ", min_ms=" << min_ms
                  << ", gpu_sum=" << gpu_out
                  << ", match=" << (mismatch ? "false" : "true")
                  << std::endl;
    }

    cudaFree(input_device);
    cudaFree(partial_output_device);
    cudaFree(scratch_output_device);
    cudaFree(final_output_device);
    free(input_host);
    cudaStreamDestroy(stream);

    return 0;
}