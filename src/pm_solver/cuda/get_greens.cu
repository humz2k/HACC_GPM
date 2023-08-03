#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include "calcGreens.hpp"

#define VerboseGreens

template<class T>
__global__ void getGreensKernel(T* __restrict d_greens, int ng, int np)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= (ng*ng*ng))return;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    d_greens[idx] = calcGreens(idx3d,ng,np);

}

template<class T>
__global__ void getGreensParallelKernel(T* __restrict d_greens, int ng, int3 local_grid_size, int3 local_coords)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int n = local_grid_size.x*local_grid_size.y*local_grid_size.z;
    if (idx >= n)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    d_greens[idx] = calcGreens(idx3d,ng,ng);

}

template<class T>
CPUTimer_t launch_getgreens(T* d_greens, int ng, int np, int numBlocks, int blockSize, int calls){

    getIndent(calls);

    return InvokeGPUKernel(getGreensKernel,numBlocks,blockSize,d_greens,ng,np);

}

template CPUTimer_t launch_getgreens<double>(double*,int,int,int,int,int);
template CPUTimer_t launch_getgreens<float>(float*,int,int,int,int,int);

template<class T>
CPUTimer_t launch_getgreens(T* d_greens, int ng, int nlocal, int3 local_grid_size_vec, int3 grid_coords_vec, int world_rank, int numBlocks, int blockSize,  int calls){

    getIndent(calls);

    return InvokeGPUKernelParallel(getGreensParallelKernel,numBlocks,blockSize,d_greens,ng,local_grid_size_vec,grid_coords_vec);

}

template CPUTimer_t launch_getgreens<double>(double*,int,int,int3,int3,int,int,int,int);
template CPUTimer_t launch_getgreens<float>(float*,int,int,int3,int3,int,int,int,int);
