#include "../ic_kernels.hpp"
#include <curand.h>
#include <curand_kernel.h>

__global__ void initRNG(curandState *state, int seed){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);

}

__global__ void initRNG(curandState *state, int seed, int nlocal, int ng, int3 local_grid_size, int3 local_coords){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx >= nlocal)return;
  int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
  int global_idx = idx3d.x * ng * ng + idx3d.y * ng + idx3d.z;
  curand_init(seed, global_idx, 0, &state[idx]);

}

template<class T>
__global__ void GenerateRealRandom(curandState* state, T* __restrict grid, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t amp = curand_normal_double(state + idx);
    T out;
    out.x = amp;
    out.y = 0;
    grid[idx] = out;
}

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    curandState* rngState; cudaCall(cudaMalloc,&rngState,sizeof(curandState)*ng*ng*ng);
    InvokeGPUKernel(initRNG,numBlocks,blockSize,rngState,seed);
    InvokeGPUKernel(GenerateRealRandom,numBlocks,blockSize,rngState,d_grid1,ng*ng*ng);
    cudaCall(cudaFree,rngState);
}

void launch_generate_rng(floatFFT_t* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    curandState* rngState; cudaCall(cudaMalloc,&rngState,sizeof(curandState)*ng*ng*ng);
    InvokeGPUKernel(initRNG,numBlocks,blockSize,rngState,seed);
    InvokeGPUKernel(GenerateRealRandom,numBlocks,blockSize,rngState,d_grid1,ng*ng*ng);
    cudaCall(cudaFree,rngState);
}

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int nlocal, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    curandState* rngState; cudaCall(cudaMalloc,&rngState,sizeof(curandState)*nlocal);
    InvokeGPUKernelParallel(initRNG,numBlocks,blockSize,rngState,seed,nlocal,ng,local_grid_size,local_coords);
    InvokeGPUKernelParallel(GenerateRealRandom,numBlocks,blockSize,rngState,d_grid1,nlocal);
    cudaCall(cudaFree,rngState);
}