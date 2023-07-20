#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

#define VerboseSolver

__device__ __forceinline__ float get_gradient(float kmode){
    return sinf(kmode);
}

__global__ void kspace_solve(deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    hostFFT_t greens = __ldg(&d_greens[idx]);
    deviceFFT_t rho = __ldg(&d_rho[idx]);
    rho.x *= greens;
    rho.y *= greens;
    d_rho[idx] = rho;
}

__global__ void kspace_solve_gradient(deviceFFT_t* __restrict d_x, deviceFFT_t* __restrict d_y, deviceFFT_t* __restrict d_z, const deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens, float b1, float b2, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    hostFFT_t greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    deviceFFT_t rho = __ldg(&d_rho[idx]);

    deviceFFT_t out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    deviceFFT_t out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    deviceFFT_t out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

__global__ void combine(float4* __restrict out, const deviceFFT_t* __restrict d_x, const deviceFFT_t* __restrict d_y, const deviceFFT_t* __restrict d_z){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    deviceFFT_t x = __ldg(&d_x[idx]);
    deviceFFT_t y = __ldg(&d_y[idx]);
    deviceFFT_t z = __ldg(&d_z[idx]);

    float4 this_out;
    this_out.x = x.x;
    this_out.y = y.x;
    this_out.z = z.x;

    out[idx] = this_out;
}

void HACCGPM::serial::Solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseSolver
    printf("%sSolve was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Doing forward fft...\n",indent);
    #endif
    HACCGPM::serial::forward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done forward fft.\n",indent);
    printf("%s   Calling kspace_solve...\n",indent);
    #endif
    InvokeGPUKernel(kspace_solve,numBlocks,blockSize,d_rho,d_greens);
    #ifdef VerboseSolver
    printf("%s      Called kspace_solve.\n",indent);
    printf("%s   Doing backward fft...\n",indent);
    #endif
    HACCGPM::serial::backward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done backward fft.\n",indent);
    #endif
}

void HACCGPM::serial::SolveGradient(float4* d_grad, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseSolver
    printf("%sSolveGradient was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Doing forward fft...\n",indent);
    #endif
    HACCGPM::serial::forward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done forward fft.\n",indent);
    printf("%s   Allocating d_x, d_y, d_z...\n",indent);
    #endif
    deviceFFT_t* d_x; cudaCall(cudaMalloc,&d_x,sizeof(deviceFFT_t)*ng*ng*ng);
    deviceFFT_t* d_y; cudaCall(cudaMalloc,&d_y,sizeof(deviceFFT_t)*ng*ng*ng);
    deviceFFT_t* d_z; cudaCall(cudaMalloc,&d_z,sizeof(deviceFFT_t)*ng*ng*ng);
    #ifdef VerboseSolver
    printf("%s      Allocated d_x, d_y, d_z\n",indent);
    printf("%s   Calling kspace_solve_gradient...\n",indent);
    #endif

    double const b1 =   4.0 / 3.0;
    double const b2 = - 1.0 / 6.0;

    InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,b1,b2,ng);

    #ifdef VerboseSolver
    printf("%s      Called kspace_solve_gradient.\n",indent);
    printf("%s   Doing backward ffts...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(d_x,ng,calls+1);
    HACCGPM::serial::backward_fft(d_y,ng,calls+1);
    HACCGPM::serial::backward_fft(d_z,ng,calls+1);

    //SHOULD WE NORMALIZE THIS???

    #ifdef VerboseSolver
    printf("%s      Done backward ffts.\n",indent);
    printf("%s   Calling combine...\n",indent);
    #endif

    InvokeGPUKernel(combine,numBlocks,blockSize,d_grad,d_x,d_y,d_z);

    #ifdef VerboseSolver
    printf("%s      Called combine.\n",indent);
    printf("%s   Freeing d_x, d_y, d_z...\n",indent);
    #endif

    cudaCall(cudaFree,d_x);
    cudaCall(cudaFree,d_y);
    cudaCall(cudaFree,d_z);

    #ifdef VerboseSolver
    printf("%s      Freed d_x, d_y, d_z.\n",indent);
    #endif
}