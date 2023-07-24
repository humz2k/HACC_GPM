#include "../pm_kernels.hpp"

__global__ void UpdatePosKernel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, float ng){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_pos = __ldg(&d_pos[idx]);
    float4 my_vel = __ldg(&d_vel[idx]);
    my_pos.x += my_vel.x * prefactor;
    my_pos.y += my_vel.y * prefactor;
    my_pos.z += my_vel.z * prefactor;
    my_pos.x = fmod(my_pos.x + ng,ng);
    my_pos.y = fmod(my_pos.y + ng,ng);
    my_pos.z = fmod(my_pos.z + ng,ng);
    //if ((my_pos.x < 0 || my_pos.x >= ng) || (my_pos.y < 0 || my_pos.y >= ng) || (my_pos.z < 0 || my_pos.z >= ng)){
    //    printf("%g %g %g\n",my_pos.x,my_pos.y,my_pos.z);
    //    printf("FUCK!!!\n");
    //}
    d_pos[idx] = my_pos;
}

__global__ void UpdatePosKernelParallel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n)return;
    float4 my_pos = __ldg(&d_pos[idx]);
    if (my_pos.w < -1)return;
    float4 my_vel = __ldg(&d_vel[idx]);
    my_pos.x += my_vel.x * prefactor;
    my_pos.y += my_vel.y * prefactor;
    my_pos.z += my_vel.z * prefactor;
    d_pos[idx] = my_pos;
}

CPUTimer_t launch_updatepos(float4* d_pos, float4* d_vel, float prefactor, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(UpdatePosKernel,numBlocks,blockSize,d_pos,d_vel,prefactor,(float)ng);
}

CPUTimer_t launch_updatepos(float4* d_pos, float4* d_vel, float prefactor, int n_particles, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(UpdatePosKernelParallel,numBlocks,blockSize,d_pos,d_vel,prefactor,n_particles);   
}