#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

__global__ void set_invalid(float4* __restrict d_pos, int mem_frac){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= mem_frac) return;
    d_pos[idx] = make_float4(0,0,0,-10);
}

HACCGPM::parallel::MemoryManager::MemoryManager(HACCGPM::Params params){
    world_rank = params.world_rank;
    if (params.world_rank == 0)printf("MemoryManager:\n   Allocating d_vel,d_pos,d_greens,d_grid,d_grad,d_tempgrid...\n");
    int mem_frac = params.nlocal * params.frac;
    if (params.world_rank == 0)printf("   Mem frac: %d / %d\n",mem_frac,params.nlocal);
    cudaCall(cudaMalloc,&d_vel,sizeof(float4)*mem_frac);
    if (params.world_rank == 0)printf("   Allocated d_vel: %lu bytes.\n",sizeof(float4)*mem_frac);

    cudaCall(cudaMalloc,&d_pos,sizeof(float4)*mem_frac);
    if (params.world_rank == 0)printf("   Allocated d_pos: %lu bytes.\n",sizeof(float4)*mem_frac);

    int blockSize = params.blockSize;
    int numBlocks = (mem_frac + (blockSize - 1)) / blockSize;
    getIndent(0);
    InvokeGPUKernelParallel(set_invalid,numBlocks,blockSize,d_pos,mem_frac);
    if (params.world_rank == 0)printf("   set_invalid d_pos\n");

    cudaCall(cudaMalloc,&d_greens,sizeof(hostFFT_t)*params.nlocal);
    if (params.world_rank == 0)printf("   Allocated d_greens: %lu bytes.\n",sizeof(hostFFT_t)*params.nlocal);

    cudaCall(cudaMalloc,&d_grid,sizeof(deviceFFT_t)*params.nlocal);
    if (params.world_rank == 0)printf("   Allocated d_grid: %lu bytes.\n",sizeof(deviceFFT_t)*params.nlocal);

    double ol = params.ol;
    if (params.world_rank == 0)printf("   OL (mpc) = %g\n",ol);
    int overload = params.overload;
    if (params.world_rank == 0)printf("   overload (grid) = %d\n",overload);

    int x = params.local_grid_size[0] + overload*2;
    int y = params.local_grid_size[1] + overload*2;
    int z = params.local_grid_size[2] + overload*2;

    if (params.world_rank == 0)printf("   overload volume = (%d %d %d) + 2*(%d %d %d) = (%d %d %d)\n",params.local_grid_size[0],
                                                                                    params.local_grid_size[1],
                                                                                    params.local_grid_size[2],
                                                                                    overload,
                                                                                    overload,
                                                                                    overload,
                                                                                    x,y,z);

    int extra_grid_size = (x)*(y)*(z);
    cudaCall(cudaMalloc,&d_tempgrid,sizeof(float)*extra_grid_size);
    if (params.world_rank == 0)printf("   Allocated d_tempgrid: %lu bytes.\n",sizeof(float)*extra_grid_size);

    //cudaCall(cudaMalloc,&d_extragrid,sizeof(deviceFFT_t)*(params.ng));

    //cudaCall(cudaMalloc,&d_grid2,sizeof(deviceFFT_t)*params.nlocal);
    //if (params.world_rank == 0)printf("   Allocated d_grid: %lu bytes.\n",sizeof(deviceFFT_t)*params.nlocal);

    cudaCall(cudaMalloc,&d_grad,sizeof(float4)*mem_frac);
    if (params.world_rank == 0)printf("   Allocated d_grad: %lu bytes.\n",sizeof(float4)*mem_frac);
}

HACCGPM::parallel::MemoryManager::~MemoryManager(){
    if (world_rank == 0)printf("MemoryManager:\n   Freeing d_vel,d_pos,d_greens,d_grid,d_grad,d_tempgrid...\n");
    
    cudaCall(cudaFree,d_pos);
    cudaCall(cudaFree,d_vel);
    cudaCall(cudaFree,d_greens);
    cudaCall(cudaFree,d_grid);
    cudaCall(cudaFree,d_tempgrid);
    cudaCall(cudaFree,d_grad);

    if (world_rank == 0)printf("      Freed d_vel,d_pos,d_greens,d_grid,d_grad,d_tempgrid.\n");
}

HACCGPM::serial::MemoryManager::MemoryManager(HACCGPM::Params params){
    printf("MemoryManager:\n   Allocating d_vel,d_pos,d_greens,d_grid,d_tempgrid,d_grad...\n");
    
    cudaCall(cudaMalloc,&d_vel,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_vel: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_pos,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_pos: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_greens,sizeof(hostFFT_t)*params.ng*params.ng*params.ng);
    printf("   Allocated d_greens: %lu bytes.\n",sizeof(hostFFT_t)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_grid,sizeof(deviceFFT_t)*params.ng*params.ng*params.ng);
    printf("   Allocated d_grid: %lu bytes.\n",sizeof(deviceFFT_t)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_tempgrid,sizeof(float)*params.ng*params.ng*params.ng);
    printf("   Allocated d_tempgrid: %lu bytes.\n",sizeof(float)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_grad,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_grad: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);
}

HACCGPM::serial::MemoryManager::~MemoryManager(){
    printf("MemoryManager:\n   Freeing d_vel,d_pos,d_greens,d_grid,d_tempgrid,d_grad...\n");
    
    cudaCall(cudaFree,d_pos);
    cudaCall(cudaFree,d_vel);
    cudaCall(cudaFree,d_greens);
    cudaCall(cudaFree,d_grid);
    cudaCall(cudaFree,d_tempgrid);
    cudaCall(cudaFree,d_grad);

    printf("      Freed d_vel,d_pos,d_greens,d_grid,temp_grid,d_grad.\n");
}