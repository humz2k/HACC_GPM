#include "../ic_kernels.hpp"

__global__ void ScaleAmplitudes(deviceFFT_t* __restrict grid, const hostFFT_t* __restrict scale, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    deviceFFT_t current = __ldg(&grid[idx]);
    current.x *= scale_by;
    current.y *= scale_by;
    grid[idx] = current;
}

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    InvokeGPUKernelParallel(ScaleAmplitudes,numBlocks,blockSize,grid,scale,nlocal);
}

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls){
    launch_scale_amplitudes(grid,scale,nlocal,0,numBlocks,blockSize,calls);
}

__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    deviceFFT_t old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}