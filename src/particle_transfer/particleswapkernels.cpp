#include "kernels.hpp"

//#define VerboseSwap

CPUTimer_t HACCGPM::parallel::insertParticles(float4* d_pos, float4* d_vel, float4* h_swap, int n_new, int n_particles, int blockSize, int world_rank, int calls){
    int numBlocks = (n_particles + blockSize - 1) / blockSize;
    getIndent(calls);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%sinsertParticles was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    if(world_rank == 0)printf("%s   Copying to device\n",indent);
    #endif

    CPUTimer_t start = CPUTimer();

    float4* d_swap; cudaCall(cudaMalloc,&d_swap,sizeof(float4)*2*n_new);
    cudaCall(cudaMemcpy,d_swap,h_swap,sizeof(float4)*2*n_new,cudaMemcpyHostToDevice);

    int* d_count; cudaCall(cudaMalloc,&d_count,sizeof(int)*1);
    cudaCall(cudaMemset,d_count,0,sizeof(int)*1);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%s      Copied to device\n",indent);
    #endif
    CPUTimer_t gpu_time = 0;
    gpu_time += launch_combine_particles(d_pos,d_vel,d_swap,d_count,n_new,n_particles,world_rank,numBlocks,blockSize,calls);

    cudaCall(cudaFree,d_swap);
    cudaCall(cudaFree,d_count);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t total_time = end-start;

    if(world_rank == 0)printf("%s   insertParticles took %llu us\n",indent,total_time);
    return gpu_time;
}

CPUTimer_t HACCGPM::parallel::LoadIntoBuffers(float4* h_swap, int* n_swaps, int* h_starts, float4* d_pos, float4* d_vel, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int n_particles, int ng, int blockSize, int world_rank, int world_size, int calls){
    int numBlocks = (n_particles + blockSize - 1) / blockSize;

    getIndent(calls);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%sLoadIntoBuffers was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif

    CPUTimer_t start = CPUTimer();
    CPUTimer_t gpu_time = 0;

    int* d_ns; cudaCall(cudaMalloc,&d_ns,sizeof(int)*world_size);
    int* d_count; cudaCall(cudaMalloc,&d_count,sizeof(int)*world_size);
    cudaCall(cudaMemset,d_ns,0,sizeof(int) * world_size);
    cudaCall(cudaMemset,d_count,0,sizeof(int) * world_size);
   
    gpu_time += launch_count_buffer(d_ns,d_pos,n_particles,ng,local_grid_size,local_coords,dims,world_rank,numBlocks,blockSize,calls);

    cudaCall(cudaMemcpy, n_swaps, d_ns, sizeof(int)*world_size, cudaMemcpyDeviceToHost);
    
    h_starts[0] = 0;
    int transfer_size = 0;
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            transfer_size += n_swaps[i];
        }
        if (i != (world_size - 1))h_starts[i+1] = transfer_size;
    }
    int* d_starts; cudaCall(cudaMalloc,&d_starts,sizeof(int)*world_size);

    cudaCall(cudaMemcpy,d_starts,h_starts,sizeof(int)*world_size,cudaMemcpyHostToDevice);

    float4* d_swap; cudaCall(cudaMalloc,&d_swap,sizeof(float4)*2*transfer_size);

    gpu_time += launch_load_buffer(d_swap,d_starts,d_count,d_pos,d_vel,n_particles,ng,local_grid_size,local_coords,dims,world_rank,numBlocks,blockSize,calls);

    cudaCall(cudaMemcpy, h_swap, d_swap, sizeof(float4)*2*transfer_size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_ns);
    cudaCall(cudaFree,d_count);
    cudaCall(cudaFree,d_swap);
    cudaCall(cudaFree,d_starts);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t total_time = end-start;

    if(world_rank == 0)printf("%s   LoadIntoBuffers took %llu us\n",indent,total_time);
    return gpu_time;
}