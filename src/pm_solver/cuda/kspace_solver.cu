#include "calcGreens.hpp"

__device__ __forceinline__ float get_gradient(float kmode){
    return sinf(kmode);
}

template<class T1, class T2>
__global__ void kspace_solve(T1* __restrict d_rho, const T2* __restrict d_greens){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    T2 greens = __ldg(&d_greens[idx]);
    T1 rho = __ldg(&d_rho[idx]);
    rho.x *= greens;
    rho.y *= greens;
    d_rho[idx] = rho;
}

template<class T, class T1>
__global__ void kspace_solve_gradient(T* __restrict d_grid, const float4* __restrict d_grad, const T1* __restrict d_greens, int ng, int dim){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    T1 greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    float4 all = __ldg(&d_grad[idx]);
    float2 rho = make_float2(all.z,all.w);

    T out;

    if (dim == 0){
        out.x = -c.x * rho.y;
        out.y = c.x * rho.x;
    } else if (dim == 1){
        out.x = -c.y * rho.y;
        out.y = c.y * rho.x;
    } else if (dim == 2){
        out.x = -c.z * rho.y;
        out.y = c.z * rho.x;
    }

    d_grid[idx] = out;
}

template<class T>
__global__ void kspace_solve_gradient(T* __restrict d_grid, const float4* __restrict d_grad, int ng, int np, int dim){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double greens = calcGreens(idx3d,ng,np);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    float4 all = __ldg(&d_grad[idx]);
    float2 rho = make_float2(all.z,all.w);

    T out;

    if (dim == 0){
        out.x = -c.x * rho.y;
        out.y = c.x * rho.x;
    } else if (dim == 1){
        out.x = -c.y * rho.y;
        out.y = c.y * rho.x;
    } else if (dim == 2){
        out.x = -c.z * rho.y;
        out.y = c.z * rho.x;
    }

    d_grid[idx] = out;
}

template<class T, class T1>
__global__ void kspace_solve_gradient(T* __restrict d_x, T* __restrict d_y, T* __restrict d_z, const T* __restrict d_rho, const T1* __restrict d_greens, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    T1 greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    T rho = __ldg(&d_rho[idx]);

    T out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    T out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    T out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

template<class T>
__global__ void kspace_solve_gradient(T* __restrict d_x, T* __restrict d_y, T* __restrict d_z, const T* __restrict d_rho, int ng, int np){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double greens = calcGreens(idx3d,ng,np);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    T rho = __ldg(&d_rho[idx]);

    T out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    T out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    T out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

template<class T1, class T2>
__global__ void kspace_solve_gradient_parallel(T1* __restrict d_x, T1* __restrict d_y, T1* __restrict d_z, const T1* __restrict d_rho, const T2* __restrict d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    T2 greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    T1 rho = __ldg(&d_rho[idx]);

    T1 out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    T1 out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    T1 out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

template<class T1, class T2>
CPUTimer_t launch_kspace_solve(T1* d_rho, T2* d_greens, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve,numBlocks,blockSize,d_rho,d_greens);
}

template CPUTimer_t launch_kspace_solve<deviceFFT_t,hostFFT_t>(deviceFFT_t*,hostFFT_t*,int,int,int);
template CPUTimer_t launch_kspace_solve<deviceFFT_t,float>(deviceFFT_t*,float*,int,int,int);
template CPUTimer_t launch_kspace_solve<floatFFT_t,hostFFT_t>(floatFFT_t*,hostFFT_t*,int,int,int);
template CPUTimer_t launch_kspace_solve<floatFFT_t,float>(floatFFT_t*,float*,int,int,int);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_grid, float4* d_grad, T2* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,d_greens,ng,dim);
}

template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,hostFFT_t>(deviceFFT_t*,float4*,hostFFT_t*,int,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,float>(deviceFFT_t*,float4*,float*,int,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,hostFFT_t>(floatFFT_t*,float4*,hostFFT_t*,int,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,float>(floatFFT_t*,float4*,float*,int,int,int,int,int);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_x, T1* d_y, T1* d_z, T1* d_rho, T2* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}

template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,hostFFT_t>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,hostFFT_t*,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,float>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,float*,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,hostFFT_t>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,hostFFT_t*,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,float>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,float*,int,int,int,int);

template<class T>
CPUTimer_t launch_kspace_solve_gradient(T* d_grid, float4* d_grad, int dim, int ng, int np, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,ng,np,dim);
}

template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t>(deviceFFT_t*,float4*,int,int,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t>(floatFFT_t*,float4*,int,int,int,int,int,int);

template<class T>
CPUTimer_t launch_kspace_solve_gradient(T* d_x, T* d_y, T* d_z, T* d_rho, int ng, int np, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,ng,np);
}

template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,int,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,int,int,int,int,int);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_x, T1* d_y, T1* d_z, T1* d_rho, T2* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}

template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,hostFFT_t>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,hostFFT_t*,int,int,int,int3,int3,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<deviceFFT_t,float>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,float*,int,int,int,int3,int3,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,hostFFT_t>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,hostFFT_t*,int,int,int,int3,int3,int,int,int,int);
template CPUTimer_t launch_kspace_solve_gradient<floatFFT_t,float>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,float*,int,int,int,int3,int3,int,int,int,int);