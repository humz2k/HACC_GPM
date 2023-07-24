
#include "../kernels.hpp"

__device__ __forceinline__ float get_gradient(float kmode){
    return sinf(kmode);
}

__global__ void kspace_solve(deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    hostFFT_t greens = __ldg(&d_greens[idx]);
    deviceFFT_t rho = __ldg(&d_rho[idx]);
    rho.x *= greens;
    rho.y *= greens;
    d_rho[idx] = rho;
}

__global__ void kspace_solve_gradient(deviceFFT_t* __restrict d_x, deviceFFT_t* __restrict d_y, deviceFFT_t* __restrict d_z, const deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    hostFFT_t greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    deviceFFT_t rho = __ldg(&d_rho[idx]);

    deviceFFT_t out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    deviceFFT_t out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    deviceFFT_t out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

__global__ void kspace_solve_gradient_parallel(deviceFFT_t* __restrict d_x, deviceFFT_t* __restrict d_y, deviceFFT_t* __restrict d_z, const deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    hostFFT_t greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    deviceFFT_t rho = __ldg(&d_rho[idx]);

    deviceFFT_t out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    deviceFFT_t out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    deviceFFT_t out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

CPUTimer_t launch_kspace_solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve,numBlocks,blockSize,d_rho,d_greens);
}

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}