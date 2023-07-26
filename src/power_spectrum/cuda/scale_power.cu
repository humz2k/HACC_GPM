#include "power_kernels.hpp"

__global__ void scalePower(deviceFFT_t* __restrict data, double ng, double rl, int nlocal){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    double scale = (1.0f/(ng*ng*ng)) * (rl/ng);
    deviceFFT_t old = __ldg(&data[idx]);
    old.x = (old.x * old.x + old.y * old.y) * scale;
    data[idx] = old;

}

__global__ void scalePower(floatFFT_t* __restrict data, double ng, double rl, int nlocal){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    double scale = (1.0f/(ng*ng*ng)) * (rl/ng);
    floatFFT_t old = __ldg(&data[idx]);
    old.x = (old.x * old.x + old.y * old.y) * scale;
    data[idx] = old;

}