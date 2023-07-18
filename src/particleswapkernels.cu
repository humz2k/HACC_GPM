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

__global__ void init_count_buffer(int* ns, const float4* __restrict d_pos, int n_particles, int3 local_grid_size){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;
    for (int i = 0; i < 27; i++){
        int x = (i/3)/3;
        int y = (i - x*3*3)/3;
        int z = i - x*3*3 - y*3;
        x = (x-1)*local_grid_size.x;
        y = (y-1)*local_grid_size.y;
        z = (z-1)*local_grid_size.z;
        float3 tmp_particle = make_float3(my_particle.x - x, my_particle.y - y, my_particle.z - z);
        if ((tmp_particle.x >= 0 && tmp_particle.x < local_grid_size.x) && 
            (tmp_particle.y >= 0 && tmp_particle.y < local_grid_size.y) &&
            (tmp_particle.z >= 0 && tmp_particle.z < local_grid_size.z)){
                //printf("Particle %g %g %g %g going to idx %d\n",my_particle.x,my_particle.y,my_particle.z,my_particle.w,i);
                atomicAdd(&ns[i],1);
            }

    }
}

__global__ void init_load_buffer(float4* __restrict d_pos, float4* __restrict d_vel, float4* __restrict d_pos_swap, float4* __restrict d_vel_swap, int* count, int* starts, const float4* __restrict d_tmppos, const float4* __restrict d_tmpvel, int3 local_grid_size, int n_particles, int world_rank){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_tmppos[idx]);
    if (my_particle.w < -1)return;
    float4 my_vel = __ldg(&d_tmpvel[idx]);
    for (int i = 0; i < 27; i++){
        int x = (i/3)/3;
        int y = (i - x*3*3)/3;
        int z = i - x*3*3 - y*3;
        x = (x-1)*local_grid_size.x;
        y = (y-1)*local_grid_size.y;
        z = (z-1)*local_grid_size.z;
        float4 tmp_particle = make_float4(my_particle.x - x, my_particle.y - y, my_particle.z - z, my_particle.w);
        if ((tmp_particle.x >= 0 && tmp_particle.x < local_grid_size.x) && 
            (tmp_particle.y >= 0 && tmp_particle.y < local_grid_size.y) &&
            (tmp_particle.z >= 0 && tmp_particle.z < local_grid_size.z)){
                //printf("Particle %g %g %g %g going to idx %d\n",my_particle.x,my_particle.y,my_particle.z,my_particle.w,i);
                float4* pos_dest;
                float4* vel_dest;
                pos_dest = d_pos;
                vel_dest = d_vel;
                if (i != 13){
                    int start = starts[i];
                    pos_dest = &d_pos_swap[start];
                    vel_dest = &d_vel_swap[start];
                }
                int j = atomicAdd(&count[i],1);
                pos_dest[j] = tmp_particle;
                vel_dest[j] = my_vel;
                /*if (i > 13){
                    if (world_rank == 0){
                        printf("i=%d j=%d [%g %g %g %g]\n",i,j,pos_dest[j].x,pos_dest[j].y,pos_dest[j].z,pos_dest[j].w);
                    }
                }*/
            }

    }
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

__global__ void loadGridBuffersKernel(float* __restrict d_out, const float* __restrict d_grid, int3 local_grid_size, int n_extra){
    int idx = threadIdx.x+blockDim.x+blockIdx.x;
    if (idx >= n_extra)return;
    //int3 start = make_int3(0,0,0);
    int x = local_grid_size.x;
    int y = local_grid_size.y;
    int z = local_grid_size.z;
    int tmp_idx = idx;

    for (int i = 1; i < 8; i++){
        int l = i/4;
        int m = (i - l*4)/2;
        int n = (i - l*4) - m*2;

        int l_mul = ((l+1)%2)*x;
        int m_mul = ((m+1)%2)*y;
        int n_mul = ((n+1)%2)*z;
        if (l_mul == 0){
            l_mul = 1;
        }
        if (m_mul == 0){
            m_mul == 1;
        }
        if (n_mul == 0){
            n_mul == 1;
        }

        int bin_size = l_mul*m_mul*n_mul;

        if (tmp_idx < bin_size){
            ///THIS IS WRONG!!!! PLEASE FIX. THE INDEXES ARE NOT ALWAYS CONTIGUOUS...
            int global_index = x * l * (y+1)*(z+1) + y * m * (z+1) + (z+1) * n + tmp_idx;
            float my_cell = __ldg(&d_grid[global_index]);
            d_out[idx] = my_cell;
            return;
        }

        tmp_idx -= bin_size;

    }

    /*if (0 <= tmp_idx < z*y){
        start.z = z;
    } else{
        tmp_idx -= z*y;
        if (0 <= tmp_idx < y*x){
            start.y = y;
        } else {
            tmp_idx -= y*x;
            if (0 <= tmp_idx < y){
                start.y = y;
                start.z = z;
            } else {
                tmp_idx -= y;
                if (0 <= tmp_idx < z*x){
                    start.x = x;
                } else {
                    tmp_idx -= z*x;
                    if (0 <= tmp_idx < z){
                        start.x = x;
                        start.z = z;
                    } else {
                        tmp_idx -= z;
                        if (0 <= tmp_idx < x){
                            start.x = x;
                            start.y = y;
                        } else{
                            start.x = x;
                            start.y = y;
                            start.z = z;
                        }}}}}}*/

    /*int global_index = start.x * (y+1)*(z+1) + start.y * (z+1) + start.z + 1 + tmp_idx;
    float my_cell = __ldg(&d_grid[global_index]);
    d_out[idx] = my_cell;*/

}

CPUTimer_t HACCGPM::parallel::loadGridBuffers(float* d_extragrid, float* h_transfer, int3 local_grid_size, int blockSize, int world_rank, int calls){
    int n_extra = (local_grid_size.x+1)*(local_grid_size.y+1)*(local_grid_size.z+1) - (local_grid_size.x*local_grid_size.y*local_grid_size.z);
    int numBlocks = (n_extra + (blockSize - 1))/blockSize;

    getIndent(calls);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%sloadGridBuffers was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    
    float* d_transfer; cudaCall(cudaMalloc,&d_transfer,sizeof(float)*n_extra);

    CPUTimer_t gpu_time = InvokeGPUKernelParallel(loadGridBuffersKernel,numBlocks,blockSize,d_transfer,d_extragrid,local_grid_size,n_extra);

    cudaCall(cudaMemcpy,h_transfer,d_transfer,sizeof(float)*n_extra,cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_transfer);

    return gpu_time;
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

CPUTimer_t HACCGPM::parallel::initLoadIntoBuffers(float4** swap_pos, float4** swap_vel, int* n_swaps,float4* d_pos, float4* d_vel, int nlocal, int3 local_grid_size, int n_particles, int blockSize, int world_rank, int calls){
    int numBlocks = (n_particles + blockSize - 1) / blockSize;

    getIndent(calls);

    #ifdef VerboseSwap
    if(world_rank == 0)printf("%sinitLoadIntoBuffers was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif

    CPUTimer_t start = CPUTimer();
    CPUTimer_t gpu_time = 0;

    int* d_ns; cudaCall(cudaMalloc,&d_ns,sizeof(int)*27);
    cudaCall(cudaMemset,d_ns,0,sizeof(int)*27);
    int* h_ns = (int*)malloc(sizeof(int) * 27);
    gpu_time += InvokeGPUKernelParallel(init_count_buffer,numBlocks,blockSize,d_ns,d_pos,n_particles,local_grid_size);
    cudaCall(cudaMemcpy, h_ns, d_ns, sizeof(int)*27, cudaMemcpyDeviceToHost);
    int tmp_size = 0;
    int* h_starts = (int*)malloc(sizeof(int) * 27);
    int* d_starts; cudaCall(cudaMalloc,&d_starts,sizeof(int)*27);
    h_starts[0] = 0;
    for (int i = 0; i < 27; i++){
        if (i == 13)continue;
        tmp_size += h_ns[i];
        h_starts[i] = 0;
        for (int j = 0; j < i; j++){
            if (j == 13)continue;
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

    if(world_rank == 0)printf("%s   initLoadIntoBuffers took %llu us\n",indent,total_time);
    return gpu_time;
}