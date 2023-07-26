#include "power_kernels.hpp"

__global__ void foldParticles(float4* __restrict d_pos, double ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float4 my_particle = __ldg(&d_pos[idx]);
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

__global__ void ScaleParticles(float4* __restrict d_pos, double oldNg, double newNg){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float4 my_particle = __ldg(&d_pos[idx]);
    float mul = newNg/oldNg;
    my_particle.x *= mul;
    my_particle.y *= mul;
    my_particle.z *= mul;
    d_pos[idx] = my_particle;
}

__global__ void cpy(float4* __restrict dest, const float4* __restrict source){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    dest[idx] = __ldg(&source[idx]);
}