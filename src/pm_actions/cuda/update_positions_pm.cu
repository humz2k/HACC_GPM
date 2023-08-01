#include "../pm_kernels.hpp"

template<class T>
__global__ void UpdatePosKernel(T* __restrict d_pos, const T* __restrict d_vel, float prefactor, float ng, int np){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= (np*np*np))return;
    T my_pos = __ldg(&d_pos[idx]);
    T my_vel = __ldg(&d_vel[idx]);
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

__global__ void UpdatePosKernelParallel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, int n, int* do_refresh, int3 local_grid_size, int overload){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n)return;
    float4 my_pos = __ldg(&d_pos[idx]);
    if (my_pos.w < -1)return;
    float4 my_vel = __ldg(&d_vel[idx]);
    my_pos.x += my_vel.x * prefactor;
    my_pos.y += my_vel.y * prefactor;
    my_pos.z += my_vel.z * prefactor;
    d_pos[idx] = my_pos;

    if ((my_pos.x < -overload) || 
        (my_pos.y < -overload) ||
        (my_pos.z < -overload) ||
        (my_pos.x > (local_grid_size.x - 1) + overload) ||
        (my_pos.y > (local_grid_size.y - 1) + overload) ||
        (my_pos.z > (local_grid_size.z - 1) + overload)){
        *do_refresh = 1;
    }
}

template<class T>
CPUTimer_t launch_updatepos(T* d_pos, T* d_vel, float prefactor, int ng, int np, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(UpdatePosKernel,numBlocks,blockSize,d_pos,d_vel,prefactor,(float)ng,np);
}

template CPUTimer_t launch_updatepos<float4>(float4*,float4*,float,int,int,int,int,int);
template CPUTimer_t launch_updatepos<float3>(float3*,float3*,float,int,int,int,int,int);

CPUTimer_t launch_updatepos(float4* d_pos, float4* d_vel, float prefactor, int n_particles, int3 local_grid_size, int overload, int* h_do_refresh, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    int* do_refresh; cudaCall(cudaMalloc,&do_refresh,sizeof(int));
    cudaCall(cudaMemset,do_refresh,0,sizeof(int));
    CPUTimer_t out = InvokeGPUKernelParallel(UpdatePosKernelParallel,numBlocks,blockSize,d_pos,d_vel,prefactor,n_particles,do_refresh,local_grid_size,overload);
    cudaCall(cudaMemcpy,h_do_refresh,do_refresh,sizeof(int),cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,do_refresh);
    return out;
}