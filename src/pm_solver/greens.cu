#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include "haccgpm.hpp"

#define VerboseGreens

/*__device__ __forceinline__ double get_filter(double sigma, double ns, double kmode){
    if (kmode == 0){
        return 1.0;
    }
    return exp(-0.25 * sigma * sigma * kmode * kmode) 
                * pow(sin(0.5 * kmode)/(0.5 * kmode), ns);
}*/

__device__ __forceinline__ double3 cos(float3 kmodes){
    double3 out;
    out.x = cos(kmodes.x);
    out.y = cos(kmodes.y);
    out.z = cos(kmodes.z);
    return out;
}

__forceinline__ __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__global__ void getGreens(hostFFT_t* __restrict d_greens, int ng)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0){
        d_greens[idx] = 0.0;
        return;
    }

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float3 tmpKmodes = make_float3(idx3d.x * d, idx3d.y * d, idx3d.z * d);

    double3 c = cos(tmpKmodes);

    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    double coeff = 0.5 / (ng*ng*ng);

    double out = coeff / (c.x + c.y + c.z - 3.0);

    d_greens[idx] = out;

}

__global__ void getGreensParallel(hostFFT_t* __restrict d_greens, int ng, int3 local_grid_size, int3 local_coords)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int n = local_grid_size.x*local_grid_size.y*local_grid_size.z;
    if (idx >= n)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0)){
        d_greens[idx] = 0.0;
        return;
    }

    double d = ((2*M_PI)/(((double)(ng))));

    float3 tmpKmodes = make_float3(idx3d.x * d, idx3d.y * d, idx3d.z * d);

    double3 c = cos(tmpKmodes);

    float3 kmodes = HACCGPM::parallel::get_kmodes(idx3d,ng,d);

    double coeff = 0.5 / (ng*ng*ng);

    double out = coeff / (c.x + c.y + c.z - 3.0);

    d_greens[idx] = out;

}

void HACCGPM::serial::InitGreens(hostFFT_t* d_greens, int ng, int blockSize, int calls){

    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseGreens
    printf("%sInitGreens was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Calling getGreens...\n",indent);
    #endif

    InvokeGPUKernel(getGreens,numBlocks,blockSize,d_greens,ng);

    #ifdef VerboseGreens
    printf("%s      Called getGreens...\n",indent);
    #endif

}

void HACCGPM::parallel::InitGreens(HACCGPM::parallel::MemoryManager& mem, HACCGPM::Params& params, int calls){

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