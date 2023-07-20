#include "ic_kernels.hpp"

/*__global__ void ScaleAmplitudes(deviceFFT_t* __restrict grid, const hostFFT_t* __restrict scale){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    deviceFFT_t current = __ldg(&grid[idx]);
    //printf("d_grid[%d]: %g %g\n",idx,current.x,current.y);
    current.x *= scale_by;
    current.y *= scale_by;
    //printf("%g %g\n",current.x,current.y);
    grid[idx] = current;
}*/

__global__ void ScaleAmplitudes(deviceFFT_t* __restrict grid, const hostFFT_t* __restrict scale, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    deviceFFT_t current = __ldg(&grid[idx]);
    //printf("d_grid[%d]: %g %g\n",idx,current.x,current.y);
    current.x *= scale_by;
    current.y *= scale_by;
    //printf("%g %g\n",current.x,current.y);
    grid[idx] = current;
}

/*__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    deviceFFT_t old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}*/

__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    deviceFFT_t old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}