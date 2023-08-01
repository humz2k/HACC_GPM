#include "../kernels.hpp"

template<class T>
__global__ void combine(float4* __restrict out, const T* __restrict d_x, const T* __restrict d_y, const T* __restrict d_z, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    T x = __ldg(&d_x[idx]);
    T y = __ldg(&d_y[idx]);
    T z = __ldg(&d_z[idx]);

    float4 this_out;
    this_out.x = x.x;
    this_out.y = y.x;
    this_out.z = z.x;

    out[idx] = this_out;
}

template<class T>
__global__ void copyGridToFloat4(float4* __restrict out, const T* __restrict d_grid, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ng*ng*ng)return;
    T cell = __ldg(&d_grid[idx]);

    float4 this_out;
    this_out.x = 0;
    this_out.y = 0;
    this_out.z = cell.x;
    this_out.w = cell.y;

    out[idx] = this_out;
}

__global__ void combine_parallel(float4* __restrict out, const deviceFFT_t* __restrict d_x, const deviceFFT_t* __restrict d_y, const deviceFFT_t* __restrict d_z, int3 local_grid_size, int3 local_coords, int overload, int nlocal, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;
    deviceFFT_t x = __ldg(&d_x[idx]);
    deviceFFT_t y = __ldg(&d_y[idx]);
    deviceFFT_t z = __ldg(&d_z[idx]);

    int3 globalIdx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
    int globalIdx = globalIdx3d.x * ng * ng + globalIdx3d.y * ng + globalIdx3d.z;

    float4 this_out;
    this_out.x = x.x;
    this_out.y = y.x;
    this_out.z = z.x;
    this_out.w = globalIdx;

    int3 rhoIdx3d = HACCGPM::parallel::get_local_index(idx,local_grid_size.x,local_grid_size.y,local_grid_size.z);
    rhoIdx3d.x += overload;
    rhoIdx3d.y += overload;
    rhoIdx3d.z += overload;

    int3 overload_dims = make_int3(local_grid_size.x + 2*overload, local_grid_size.y + 2*overload, local_grid_size.z + 2*overload);

    int rhoIdx = rhoIdx3d.x * overload_dims.y * overload_dims.z + rhoIdx3d.y * overload_dims.z + rhoIdx3d.z;

    out[rhoIdx] = this_out;
}

template<class T>
CPUTimer_t launch_grid2float4(float4* d_grad, T* d_grid, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(copyGridToFloat4,numBlocks,blockSize,d_grad,d_grid,ng);
}

template CPUTimer_t launch_grid2float4<deviceFFT_t>(float4*,deviceFFT_t*,int,int,int,int);
template CPUTimer_t launch_grid2float4<floatFFT_t>(float4*,floatFFT_t*,int,int,int,int);

template<class T>
CPUTimer_t launch_combine(float4* d_grad, T* d_x, T* d_y, T* d_z, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(combine,numBlocks,blockSize,d_grad,d_x,d_y,d_z,ng);
}

template CPUTimer_t launch_combine<deviceFFT_t>(float4*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,int,int,int,int);
template CPUTimer_t launch_combine<floatFFT_t>(float4*,floatFFT_t*,floatFFT_t*,floatFFT_t*,int,int,int,int);


CPUTimer_t launch_combine(float4* d_grad, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int3 local_grid_size, int3 local_coords, int overload, int nlocal, int ng, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(combine_parallel,numBlocks,blockSize,d_grad,d_x,d_y,d_z,local_grid_size,local_coords,overload,nlocal,ng);
}