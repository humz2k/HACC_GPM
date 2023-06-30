#include "stdlib.h"
#include "stdio.h"
#include "haccgpm.hpp"

HACCGPM::serial::MemoryManager::MemoryManager(HACCGPM::Params params){
    printf("MemoryManager:\n   Allocating d_vel,d_pos,d_greens,d_grid,d_grad...\n");
    
    cudaCall(cudaMalloc,&d_vel,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_vel: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_pos,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_pos: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_greens,sizeof(hostFFT_t)*params.ng*params.ng*params.ng);
    printf("   Allocated d_greens: %lu bytes.\n",sizeof(hostFFT_t)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_grid,sizeof(deviceFFT_t)*params.ng*params.ng*params.ng);
    printf("   Allocated d_grid: %lu bytes.\n",sizeof(deviceFFT_t)*params.ng*params.ng*params.ng);

    cudaCall(cudaMalloc,&d_grad,sizeof(float4)*params.ng*params.ng*params.ng);
    printf("   Allocated d_grad: %lu bytes.\n",sizeof(float4)*params.ng*params.ng*params.ng);
}

HACCGPM::serial::MemoryManager::~MemoryManager(){
    printf("MemoryManager:\n   Freeing d_vel,d_pos,d_greens,d_grid,d_grad...\n");
    
    cudaCall(cudaFree,d_pos);
    cudaCall(cudaFree,d_vel);
    cudaCall(cudaFree,d_greens);
    cudaCall(cudaFree,d_grid);
    cudaCall(cudaFree,d_grad);

    printf("      Freed d_vel,d_pos,d_greens,d_grid,d_grad.\n");
}