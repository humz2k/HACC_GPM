
#include "../kernels.hpp"

__device__ __forceinline__ float get_gradient(float kmode){
    return sinf(kmode);
}

__device__ __forceinline__ double3 cos(float3 kmodes){
    double3 out;
    out.x = cos(kmodes.x);
    out.y = cos(kmodes.y);
    out.z = cos(kmodes.z);
    return out;
}

__forceinline__ __device__ double calcGreens(int3 idx3d, int ng){

    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0))return 0.0;

    float d = ((2*M_PI)/(((float)(ng))));

    double3 c = cos(idx3d * d);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double coeff = 0.5 / (ng*ng*ng);

    double out = coeff / (c.x + c.y + c.z - 3.0);

    return out;

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
__global__ void kspace_solve_gradient(T* __restrict d_grid, const float4* __restrict d_grad, int ng, int dim){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double greens = calcGreens(idx3d,ng);

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
__global__ void kspace_solve_gradient(T* __restrict d_x, T* __restrict d_y, T* __restrict d_z, const T* __restrict d_rho, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double greens = calcGreens(idx3d,ng);

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

/*CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_grid, float4* d_grad, hostFFT_t* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,d_greens,ng,dim);
}

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_grid, float4* d_grad, float* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,d_greens,ng,dim);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_grid, float4* d_grad, hostFFT_t* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,d_greens,ng,dim);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_grid, float4* d_grad, float* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,d_greens,ng,dim);
}*/

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, float* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, floatFFT_t* d_rho, hostFFT_t* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, floatFFT_t* d_rho, float* d_greens, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng);
}






CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_grid, float4* d_grad, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,ng,dim);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_grid, float4* d_grad, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_grid,d_grad,ng,dim);
}


CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,ng);
}


CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, floatFFT_t* d_rho, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,ng);
}





CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, float* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, floatFFT_t* d_rho, hostFFT_t* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}

CPUTimer_t launch_kspace_solve_gradient(floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, floatFFT_t* d_rho, float* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords);
}