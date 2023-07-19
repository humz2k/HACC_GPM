#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"
#include "../bdwgc/include/gc.h"

#define VerboseSwap

__global__ void swap_set_invalid(float4* __restrict d_pos, int mem_frac){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= mem_frac) return;
    d_pos[idx] = make_float4(0,0,0,-10);
}

__global__ void copy(float4* __restrict dest1, const float4* __restrict source1, float4* __restrict dest2, const float4* __restrict source2, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n)return;
    float4 my_particle = __ldg(&source1[idx]);
    float4 my_vel = __ldg(&source2[idx]);
    dest1[idx] = my_particle;
    dest2[idx] = my_vel;
}

__global__ void init_count_buffer(int* ns, const float4* __restrict d_pos, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;

    my_particle.x += (float)(local_grid_size.x * local_coords.x);
    my_particle.y += (float)(local_grid_size.y * local_coords.y);
    my_particle.z += (float)(local_grid_size.z * local_coords.z);

    my_particle.x += (float)ng;
    my_particle.y += (float)ng;
    my_particle.z += (float)ng;

    my_particle.x = fmod(my_particle.x,(float)ng);
    my_particle.y = fmod(my_particle.y,(float)ng);
    my_particle.z = fmod(my_particle.z,(float)ng);

    int3 dest_coords = make_int3(my_particle.x / local_grid_size.x,my_particle.y / local_grid_size.y,my_particle.z / local_grid_size.z);

    int dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    atomicAdd(&ns[dest_rank],1);
}

__global__ void init_load_buffer(float4* d_swap, int* ns, int* counts, float4* __restrict d_pos, float4* __restrict d_vel, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;

    my_particle.x += (float)(local_grid_size.x * local_coords.x);
    my_particle.y += (float)(local_grid_size.y * local_coords.y);
    my_particle.z += (float)(local_grid_size.z * local_coords.z);

    my_particle.x += (float)ng;
    my_particle.y += (float)ng;
    my_particle.z += (float)ng;

    my_particle.x = fmod(my_particle.x,(float)ng);
    my_particle.y = fmod(my_particle.y,(float)ng);
    my_particle.z = fmod(my_particle.z,(float)ng);

    int3 dest_coords = make_int3(my_particle.x / local_grid_size.x,my_particle.y / local_grid_size.y,my_particle.z / local_grid_size.z);

    int dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    if (dest_rank == world_rank)return;

    int start = 0;
    for (int i = 0; i < dest_rank; i++){
        start += ns[i];
    }

    int count = atomicAdd(&counts[dest_rank],1);

    int indx = start + count;

    d_swap[indx*2] = my_particle;
    d_swap[indx*2+1] = __ldg(&d_vel[idx]);

    d_pos[idx] = make_float4(0,0,0,-10);
    d_vel[idx] = make_float4(0,0,0,-10);

    //FINISH ME

    //atomicAdd(&ns[dest_rank],1);
}

__global__ void findDuplicates(const float4* __restrict d_pos, float4* __restrict new_particles, int n_new, int n_particles){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    for (int i = 0; i < n_new; i++){
        float4 other_particle = __ldg(&new_particles[i]);
        if (my_particle.w == other_particle.w){
            //printf("DUPLICATE!!!\n");
            new_particles[i] = make_float4(0,0,0,-10);
            return;
        }
    }
}

__global__ void combineParticles(float4* __restrict d_pos, float4* __restrict d_vel, const float4* __restrict new_pos, const float4* __restrict new_vel, int n_new, int remaining, int n_particles){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_new)return;
    float4 my_particle = __ldg(&new_pos[idx]);
    if (my_particle.w < -1)return;
    float4 my_vel = __ldg(&new_vel[idx]);
    int dest = idx + remaining;
    d_pos[dest] = my_particle;
    d_vel[dest] = my_vel;
}

CPUTimer_t HACCGPM::parallel::insertParticles(float4* d_pos, float4* d_vel, float4* new_pos, float4* new_vel, int n_new, int remaining, int n_particles, int blockSize, int world_rank, int calls){
    int numBlocks = (n_particles + blockSize - 1) / blockSize;
    getIndent(calls);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%sinsertParticles was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    if(world_rank == 0)printf("%s   Copying to device\n",indent);
    #endif

    float4* d_new_pos; cudaCall(cudaMalloc,&d_new_pos,sizeof(float4)*n_new);
    float4* d_new_vel; cudaCall(cudaMalloc,&d_new_vel,sizeof(float4)*n_new);
    cudaCall(cudaMemcpy,d_new_pos,new_pos,sizeof(float4)*n_new,cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy,d_new_vel,new_vel,sizeof(float4)*n_new,cudaMemcpyHostToDevice);
    #ifdef VerboseSwap
    if(world_rank == 0)printf("%s      Copied to device\n",indent);
    #endif
    CPUTimer_t gpu_time = 0;
    //gpu_time += InvokeGPUKernelParallel(findDuplicates,numBlocks,blockSize,d_pos,d_new_pos,n_new,n_particles);
    gpu_time += InvokeGPUKernelParallel(combineParticles,numBlocks,blockSize,d_pos,d_vel,d_new_pos,d_new_vel,n_new,remaining,n_particles);
    return gpu_time;
}

CPUTimer_t HACCGPM::parallel::LoadIntoBuffers(float4** h_swap, int* n_swaps,float4* d_pos, float4* d_vel, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int n_particles, int ng, int blockSize, int world_rank, int world_size, int calls){
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
    //int* h_ns = (int*)malloc(sizeof(int) * world_size);
    //gpu_time += 
    InvokeGPUKernelParallel(init_count_buffer,numBlocks,blockSize,d_ns,d_pos,n_particles,ng,local_grid_size,local_coords,dims);
    cudaCall(cudaMemcpy, n_swaps, d_ns, sizeof(int)*world_size, cudaMemcpyDeviceToHost);
    
    int transfer_size = 0;
    for (int i = 0; i < world_size; i++){
        //printf("Rank %d sending %d to rank %d\n",world_rank,h_ns[i],i);
        if (i != world_rank){
            transfer_size += n_swaps[i];
        }
    }

    float4* d_swap; cudaCall(cudaMalloc,&d_swap,sizeof(float4)*2*transfer_size);

    InvokeGPUKernelParallel(init_load_buffer,numBlocks,blockSize,d_swap,d_ns,d_count,d_pos,d_vel,n_particles,ng,local_grid_size,local_coords,dims,world_rank);

    *h_swap = (float4*)malloc(sizeof(float4)*2*transfer_size);

    cudaCall(cudaMemcpy, *h_swap, d_swap, sizeof(float4)*2*transfer_size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_ns);
    cudaCall(cudaFree,d_count);
    cudaCall(cudaFree,d_swap);
    //free(h_ns);

    return 0;
    /*int tmp_size = 0;
    int* h_starts = (int*)malloc(sizeof(int) * 27);
    int* d_starts; cudaCall(cudaMalloc,&d_starts,sizeof(int)*27);
    h_starts[0] = 0;
    for (int i = 0; i < 27; i++){
        int x = (i/3)/3;
        int y = (i - x*3*3)/3;
        int z = i - x*3*3 - y*3;
        if ((x == 1) && (y == 1) && (z == 1))continue;
        tmp_size += h_ns[i];
        h_starts[i] = 0;
        for (int j = 0; j < i; j++){
            int x = (i/3)/3;
            int y = (i - x*3*3)/3;
            int z = i - x*3*3 - y*3;
            if ((x == 1) && (y == 1) && (z == 1))continue;
            h_starts[i] += h_ns[j];
        }
    }

    cudaCall(cudaMemcpy, d_starts, h_starts, sizeof(int)*27, cudaMemcpyHostToDevice);
    float4* d_pos_swap; cudaCall(cudaMalloc,&d_pos_swap,sizeof(float4)*tmp_size);
    float4* d_vel_swap; cudaCall(cudaMalloc,&d_vel_swap,sizeof(float4)*tmp_size);
    float4* d_tmppos; cudaCall(cudaMalloc,&d_tmppos,sizeof(float4)*n_particles);
    float4* d_tmpvel; cudaCall(cudaMalloc,&d_tmpvel,sizeof(float4)*n_particles);
    gpu_time += InvokeGPUKernelParallel(copy,numBlocks,blockSize,d_tmppos,d_pos,d_tmpvel,d_vel,n_particles);
    gpu_time += InvokeGPUKernelParallel(swap_set_invalid,numBlocks,blockSize,d_pos,n_particles);

    cudaCall(cudaMemset,d_ns,0,sizeof(int)*27);
    gpu_time += InvokeGPUKernelParallel(init_load_buffer,numBlocks,blockSize,d_pos,d_vel,d_pos_swap,d_vel_swap,d_ns,d_starts,d_tmppos,d_tmpvel,local_grid_size,n_particles,world_rank);

    float4* h_pos_swap = (float4*)GC_MALLOC(sizeof(float4)*tmp_size);
    float4* h_vel_swap = (float4*)GC_MALLOC(sizeof(float4)*tmp_size);
    cudaCall(cudaMemcpy, h_pos_swap, d_pos_swap, sizeof(float4)*tmp_size, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_vel_swap, d_vel_swap, sizeof(float4)*tmp_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 27; i++){
        n_swaps[i] = h_ns[i];
        swap_pos[i] = &h_pos_swap[h_starts[i]];
        swap_vel[i] = &h_vel_swap[h_starts[i]];
    }

    cudaCall(cudaFree,d_pos_swap);
    cudaCall(cudaFree,d_vel_swap);
    cudaCall(cudaFree,d_tmppos);
    cudaCall(cudaFree,d_tmpvel);
    cudaCall(cudaFree,d_ns);
    free(h_ns);
    free(h_starts);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t total_time = end-start;

    if(world_rank == 0)printf("%s   LoadIntoBuffers took %llu us\n",indent,total_time);
    return gpu_time;*/
}