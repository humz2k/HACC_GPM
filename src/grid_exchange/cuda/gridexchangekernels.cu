#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"
#include "../gridexchangekernels.hpp"

//#define VerboseGEKernels

__global__ void loadXLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,overload,ol_grid_size.y,ol_grid_size.z);

    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = idx3d.x * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadXLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;
    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading X Left Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;
    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadXLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeXLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.z);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    idx3d.x += (ol_grid_size.x - 2*overload);
    int new_idx = idx3d.x * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeXLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring X Left Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeXLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}


__global__ void loadXRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,overload,ol_grid_size.y,ol_grid_size.z);
    idx3d.x += ol_grid_size.x;
    idx3d.x -= overload;
    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = idx3d.x * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadXRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading X Right Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadXRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeXRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.z);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    //idx3d.x += ol_grid_size.x;
    //idx3d.x -= overload;
    idx3d.x += overload;
    
    //idx3d.x += (ol_grid_size.x - 2*overload);
    int new_idx = idx3d.x * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeXRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring X Right Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeXRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}

__global__ void loadYLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x - (2*overload),overload,ol_grid_size.z);

    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Y Left Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadYLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeYLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(ol_grid_size.x - (2*overload),overload,ol_grid_size.z);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    idx3d.y += (ol_grid_size.y - 2*overload);
    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeYLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Y Left Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeYLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}


__global__ void loadYRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x - (2*overload),overload,ol_grid_size.z);
    idx3d.y += ol_grid_size.y;
    idx3d.y -= overload;
    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Y Right Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadYRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeYRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(ol_grid_size.x - (2*overload),overload,ol_grid_size.z);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    //idx3d.x += ol_grid_size.x;
    idx3d.y += overload;
    
    //idx3d.x += (ol_grid_size.x - 2*overload);
    int new_idx = (idx3d.x+overload) * ol_grid_size.y * ol_grid_size.z + idx3d.y * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeYRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Y Right Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeYRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}





__global__ void loadZLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x - (2*overload),ol_grid_size.y - (2*overload),overload);

    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + (idx3d.y+overload) * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Z Left Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadZLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeZLeftKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(ol_grid_size.x - (2*overload),ol_grid_size.y - (2*overload),overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    idx3d.z += (ol_grid_size.z - 2*overload);
    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + (idx3d.y + overload) * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeZLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Z Left Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeZLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}


__global__ void loadZRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //int n = (local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload);
    if(idx >= n)return;

    //int3 ol_grid_size = make_int3(local_grid_size.x + 2*overload,local_grid_size.y + 2*overload,local_grid_size.z + 2*overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x - (2*overload),ol_grid_size.y - (2*overload),overload);
    idx3d.z += ol_grid_size.z;
    idx3d.z -= overload;
    //if (!(idx3d.x < overload))return;

    //int3 new_grid_size = make_int3(overload,ol_grid_size.y,ol_grid_size.x);

    int new_idx = (idx3d.x + overload) * ol_grid_size.y * ol_grid_size.z + (idx3d.y + overload) * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[new_idx]);
    d_out[idx] = in;

}

CPUTimer_t loadZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Z Right Buffer\n",indent);
    #endif

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    gpu_time = InvokeGPUKernelParallel(loadZRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);

    return gpu_time;
}

__global__ void storeZRightKernel(float* __restrict d_out, const float* __restrict d_in, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= n)return;
    int3 in_grid_size = make_int3(ol_grid_size.x - (2*overload),ol_grid_size.y - (2*overload),overload);
    int3 idx3d = HACCGPM::parallel::get_local_index(idx,in_grid_size.x,in_grid_size.y,in_grid_size.z);
    //idx3d.x += ol_grid_size.x;
    //idx3d.x -= overload;
    idx3d.z += overload;
    
    //idx3d.x += (ol_grid_size.x - 2*overload);
    int new_idx = (idx3d.x+overload) * ol_grid_size.y * ol_grid_size.z + (idx3d.y + overload) * ol_grid_size.z + idx3d.z;

    float in = __ldg(&d_in[idx]);
    float out = __ldg(&d_out[new_idx]);
    d_out[new_idx] = (in+out);

}

CPUTimer_t storeZRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Z Right Buffer\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    gpu_time = InvokeGPUKernelParallel(storeZRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);

    return gpu_time;
}