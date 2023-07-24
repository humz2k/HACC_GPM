#include "../kernels.hpp"

__global__ void combineParticles(float4* __restrict d_pos, float4* __restrict d_vel, const float4* __restrict d_swap, int* d_count, int n_new, int n_particles){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    if (!(my_particle.w < -1))return;
    
    int my_idx = atomicAdd(d_count,1);
    if (my_idx >= n_new)return;
    float4 new_pos = __ldg(&d_swap[my_idx*2]);
    float4 new_vel = __ldg(&d_swap[(my_idx*2) + 1]);
    d_pos[idx] = new_pos;
    d_vel[idx] = new_vel;
}

CPUTimer_t launch_combine_particles(float4* d_pos, float4* d_vel, float4* d_swap, int* d_count, int n_new, int n_particles, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(combineParticles,numBlocks,blockSize,d_pos,d_vel,d_swap,d_count,n_new,n_particles);
}