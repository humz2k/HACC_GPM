#include "../ic_kernels.hpp"

template<class T, class T1>
__global__ void copyGrid(const T* __restrict oldGrid, T1* __restrict newGrid, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ng*ng*ng)return;
    T old_cell = __ldg(&oldGrid[idx]);
    T1 new_cell;
    new_cell.x = old_cell.x;
    new_cell.y = old_cell.y;
    newGrid[idx] = new_cell;
}

template<class T, class T1>
__global__ void getRealGrid(const T* __restrict oldGrid, T1* __restrict newGrid, int dim, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ng*ng*ng)return;
    T old_cell = __ldg(&oldGrid[idx]);
    T1 new_cell = __ldg(&newGrid[idx]);
    if (dim == 0){
        new_cell.x = old_cell.x;
    } else if (dim == 1){
        new_cell.y = old_cell.x;
    } else if (dim == 2){
        new_cell.z = old_cell.x;
    }
    newGrid[idx] = new_cell;
}

template<class T>
__global__ void transformDensityField(const T* __restrict oldGrid, T* __restrict outSx, T* __restrict outSy, T* __restrict outSz, double delta, double rl, double a, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= ng*ng*ng)return;

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double k2 = kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z;

    double k2mul = (1/k2);
    if (k2 == 0){
        k2mul = 0;
    }

    double mul = (1/delta) * k2mul;

    T current = __ldg(&oldGrid[idx]);
    current.x *= mul;
    current.y *= mul;

    T sx,sy,sz;

    sx.x = current.y * kmodes.x;
    sx.y = -current.x * kmodes.x;

    sy.x = current.y * kmodes.y;
    sy.y = -current.x * kmodes.y;

    sz.x = current.y * kmodes.z;
    sz.y = -current.x * kmodes.z;

    outSx[idx] = sx;
    outSy[idx] = sy;
    outSz[idx] = sz;

}

template<class T>
__global__ void transformDensityField(const float2* __restrict oldGrid, T* __restrict out, double delta, double rl, double a, int ng, int dim){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= ng*ng*ng)return;

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double k2 = kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z;

    double k2mul = (1/k2);
    if (k2 == 0){
        k2mul = 0;
    }

    double mul = (1/delta) * k2mul;

    float2 current = __ldg(&oldGrid[idx]);
    current.x *= mul;
    current.y *= mul;

    T s;

    if(dim == 0){
        s.x = current.y * kmodes.x;
        s.y = -current.x * kmodes.x;
    } else if (dim == 1){
        s.x = current.y * kmodes.y;
        s.y = -current.x * kmodes.y;
    } else if (dim == 2){
        s.x = current.y * kmodes.z;
        s.y = -current.x * kmodes.z;
    }

    out[idx] = s;

}

__global__ void transformDensityField(const deviceFFT_t* __restrict oldGrid, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double rl, double a, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    double k2 = kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z;

    double k2mul = (1/k2);
    if (k2 == 0){
        k2mul = 0;
    }

    double mul = (1/delta) * k2mul;

    deviceFFT_t current = __ldg(&oldGrid[idx]);
    current.x *= mul;
    current.y *= mul;

    deviceFFT_t sx,sy,sz;

    sx.x = current.y * kmodes.x;
    sx.y = -current.x * kmodes.x;

    sy.x = current.y * kmodes.y;
    sy.y = -current.x * kmodes.y;

    sz.x = current.y * kmodes.z;
    sz.y = -current.x * kmodes.z;

    outSx[idx] = sx;
    outSy[idx] = sy;
    outSz[idx] = sz;

}

template<class T>
CPUTimer_t launch_transform_density_field(float2* d_grid, T* d_out, int dim, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(transformDensityField,numBlocks,blockSize,d_grid,d_out,delta,rl,1/(1+z_ini),ng,dim);
}

template CPUTimer_t launch_transform_density_field<deviceFFT_t>(float2*,deviceFFT_t*,int,double,double,double,int,int,int,int);
template CPUTimer_t launch_transform_density_field<floatFFT_t>(float2*,floatFFT_t*,int,double,double,double,int,int,int,int);

template<class T>
CPUTimer_t launch_transform_density_field(T* d_grid, T* d_x, T* d_y, T* d_z, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(transformDensityField,numBlocks,blockSize,d_grid,d_x,d_y,d_z,delta,rl,1/(1+z_ini),ng);
}

template CPUTimer_t launch_transform_density_field<deviceFFT_t>(deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,double,double,double,int,int,int,int);
template CPUTimer_t launch_transform_density_field<floatFFT_t>(floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,double,double,double,int,int,int,int);

CPUTimer_t launch_transform_density_field(deviceFFT_t* d_grid, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double rl, double z_ini, int ng, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(transformDensityField,numBlocks,blockSize,d_grid,d_x,d_y,d_z,delta,rl,1/(1+z_ini),ng,nlocal,world_rank,local_grid_size,local_coords,dims);
}

template<class T>
CPUTimer_t launch_copy_grid(T* d_grid, float2* new_grid, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(copyGrid,numBlocks,blockSize,d_grid,new_grid,ng);
}

template CPUTimer_t launch_copy_grid<deviceFFT_t>(deviceFFT_t*,float2*,int,int,int,int);
template CPUTimer_t launch_copy_grid<floatFFT_t>(floatFFT_t*,float2*,int,int,int,int);


template<class T1, class T2>
CPUTimer_t launch_get_real_grid(T1* d_grid, T2* new_grid, int dim, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(getRealGrid,numBlocks,blockSize,d_grid,new_grid,dim,ng);
}

template CPUTimer_t launch_get_real_grid<deviceFFT_t,float4>(deviceFFT_t*,float4*,int,int,int,int,int);
template CPUTimer_t launch_get_real_grid<deviceFFT_t,float3>(deviceFFT_t*,float3*,int,int,int,int,int);
template CPUTimer_t launch_get_real_grid<floatFFT_t,float4>(floatFFT_t*,float4*,int,int,int,int,int);
template CPUTimer_t launch_get_real_grid<floatFFT_t,float3>(floatFFT_t*,float3*,int,int,int,int,int);