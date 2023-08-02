#include "power_kernels.hpp"

template<class T>
__global__ void foldParticles(T* __restrict d_pos, double ng, int np){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (np*np*np))return;
    T my_particle = __ldg(&d_pos[idx]);
    my_particle.x /= ng;
    if (my_particle.x >= 0.5){
        my_particle.x -= 0.5;
    }
    my_particle.x *= 2 * ng;

    my_particle.y /= ng;
    if (my_particle.y >= 0.5){
        my_particle.y -= 0.5;
    }
    my_particle.y *= 2 * ng;

    my_particle.z /= ng;
    if (my_particle.z >= 0.5){
        my_particle.z -= 0.5;
    }
    my_particle.z *= 2 * ng;

    d_pos[idx] = my_particle;
}

template __global__ void foldParticles<float4>(float4* __restrict,double,int);
template __global__ void foldParticles<float3>(float3* __restrict,double,int);

__global__ void foldParticles(float4* __restrict d_pos, double ng, int3 local_grid_size, int3 local_coords){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float4 my_particle = __ldg(&d_pos[idx]);

    if (my_particle.w < -1)return;if (my_particle.w < -1)return;

    my_particle.x += local_grid_size.x * local_coords.x;
    my_particle.y += local_grid_size.y * local_coords.y;
    my_particle.z += local_grid_size.z * local_coords.z;
    
    my_particle.x /= ng;
    if (my_particle.x >= 0.5){
        my_particle.x -= 0.5;
    }
    my_particle.x *= 2 * ng;

    my_particle.y /= ng;
    if (my_particle.y >= 0.5){
        my_particle.y -= 0.5;
    }
    my_particle.y *= 2 * ng;

    my_particle.z /= ng;
    if (my_particle.z >= 0.5){
        my_particle.z -= 0.5;
    }
    my_particle.z *= 2 * ng;

    my_particle.x -= local_grid_size.x * local_coords.x;
    my_particle.y -= local_grid_size.y * local_coords.y;
    my_particle.z -= local_grid_size.z * local_coords.z;

    d_pos[idx] = my_particle;
}

template<class T>
__global__ void cpy(T* __restrict dest, const T* __restrict source, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)return;
    dest[idx] = __ldg(&source[idx]);
}

template __global__ void cpy<float4>(float4* __restrict, const float4* __restrict, int);
template __global__ void cpy<float3>(float3* __restrict, const float3* __restrict, int);