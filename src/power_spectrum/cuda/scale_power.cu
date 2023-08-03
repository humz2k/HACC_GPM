#include "power_kernels.hpp"

template<class T>
__global__ void scalePower(T* __restrict data, double np, double ng, double rl, int nlocal){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    double scale = (1.0f/(ng*ng*ng)) * (rl/ng);
    T old = __ldg(&data[idx]);
    old.x = (old.x * old.x + old.y * old.y) * scale;
    data[idx] = old;
}

template __global__ void scalePower<deviceFFT_t>(deviceFFT_t* __restrict,double,double,double,int);
template __global__ void scalePower<floatFFT_t>(floatFFT_t* __restrict,double,double,double,int);