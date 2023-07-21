#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include "haccgpm.hpp"

#define VerboseGreens

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

__global__ void getGreens(hostFFT_t* __restrict d_greens, int ng)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    d_greens[idx] = calcGreens(idx3d,ng);

}

__global__ void getGreensParallel(hostFFT_t* __restrict d_greens, int ng, int3 local_grid_size, int3 local_coords)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int n = local_grid_size.x*local_grid_size.y*local_grid_size.z;
    if (idx >= n)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    d_greens[idx] = calcGreens(idx3d,ng);

}

void HACCGPM::serial::InitGreens(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls){

    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;

    getIndent(calls);

    #ifdef VerboseGreens
    printf("%sInitGreens was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    printf("%s   Calling getGreens...\n",indent);
    #endif

    InvokeGPUKernel(getGreens,numBlocks,params.blockSize,mem.d_greens,params.ng);

    #ifdef VerboseGreens
    printf("%s      Called getGreens...\n",indent);
    #endif

}

void HACCGPM::parallel::InitGreens(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int calls){

    int world_rank = params.world_rank;
    int numBlocks = (params.nlocal + (params.blockSize - 1))/params.blockSize;

    getIndent(calls);

    #ifdef VerboseGreens
    if(params.world_rank == 0)printf("%sInitGreens was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    if(params.world_rank == 0)printf("%s   Calling getGreens...\n",indent);
    #endif

    InvokeGPUKernelParallel(getGreensParallel,numBlocks,params.blockSize,mem.d_greens,params.ng,params.local_grid_size_vec,params.grid_coords_vec);

    #ifdef VerboseGreens
    if(params.world_rank == 0)printf("%s      Called getGreens...\n",indent);
    #endif

}