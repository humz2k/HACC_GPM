#include "../ic_kernels.hpp"

template<class T>
__global__ void ScaleAmplitudes(T* __restrict grid, const hostFFT_t* __restrict scale, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    T current = __ldg(&grid[idx]);
    current.x *= scale_by;
    current.y *= scale_by;
    grid[idx] = current;
}

template<class T>
CPUTimer_t launch_scale_amplitudes(T* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(ScaleAmplitudes,numBlocks,blockSize,grid,scale,nlocal);
}

template CPUTimer_t launch_scale_amplitudes<deviceFFT_t>(deviceFFT_t*,hostFFT_t*,int,int,int,int,int);
template CPUTimer_t launch_scale_amplitudes<floatFFT_t>(floatFFT_t*,hostFFT_t*,int,int,int,int,int);

template<class T>
CPUTimer_t launch_scale_amplitudes(T* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls){
    return launch_scale_amplitudes(grid,scale,nlocal,0,numBlocks,blockSize,calls);
}

template CPUTimer_t launch_scale_amplitudes<deviceFFT_t>(deviceFFT_t*,hostFFT_t*,int,int,int,int);
template CPUTimer_t launch_scale_amplitudes<floatFFT_t>(floatFFT_t*,hostFFT_t*,int,int,int,int);

template<class T>
__global__ void ScaleFFT(T* __restrict data, double scale, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    T old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}

template<class T>
CPUTimer_t launch_scale_fft(T* data, double scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(ScaleFFT,numBlocks,blockSize,data,scale,nlocal);
}

template CPUTimer_t launch_scale_fft<deviceFFT_t>(deviceFFT_t*,double,int,int,int,int,int);
template CPUTimer_t launch_scale_fft<floatFFT_t>(floatFFT_t*,double,int,int,int,int,int);

template<class T>
CPUTimer_t launch_scale_fft(T* data, double scale, int nlocal, int numBlocks, int blockSize, int calls){
    return launch_scale_fft(data,scale,nlocal,0,numBlocks,blockSize,calls);
}

template CPUTimer_t launch_scale_fft<deviceFFT_t>(deviceFFT_t*,double,int,int,int,int);
template CPUTimer_t launch_scale_fft<floatFFT_t>(floatFFT_t*,double,int,int,int,int);