#include "../pm_kernels.hpp"

__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;
    float my_grid = __ldg(&d_in[idx]);
    deviceFFT_t out;
    out.x = my_grid;
    out.y = 0;
    d_out[idx] = out;
}

__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int3 local_grid_size, int overload){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;
    int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,ol_grid_size.y,ol_grid_size.z);

    idx3d.x -= overload;
    idx3d.y -= overload;
    idx3d.z -= overload;

    if (idx3d.x < 0)return;
    if (idx3d.y < 0)return;
    if (idx3d.z < 0)return;
    if (idx3d.x >= local_grid_size.x)return;
    if (idx3d.y >= local_grid_size.y)return;
    if (idx3d.z >= local_grid_size.z)return;

    float my_grid = __ldg(&d_in[idx]);
    deviceFFT_t out;
    out.x = my_grid;
    out.y = 0;
    int outidx = idx3d.x * local_grid_size.y * local_grid_size.z + idx3d.y * local_grid_size.z + idx3d.z;
    d_out[outidx] = out;
}

CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(float2complex,numBlocks,blockSize,d_out,d_in,ng*ng*ng);
}

CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int3 local_grid_size, int overload, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(float2complex,numBlocks,blockSize,d_out,d_in,local_grid_size,overload);
}