#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"
#include "../gridexchangekernels.hpp"

__global__ void loadXReturnKernel(float4* __restrict d_left, float4* __restrict d_right, const float4* __restrict d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,overload,local_grid_size.y,local_grid_size.z);

    idx3dLeft.y += overload;
    idx3dLeft.z += overload;

    int3 idx3dRight = make_int3((local_grid_size.x - idx3dLeft.x) - 1,idx3dLeft.y,idx3dLeft.z);

    idx3dLeft.x += overload;
    idx3dRight.x += overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;

    float4 left = __ldg(&d_in[idxLeft]);
    float4 right = __ldg(&d_in[idxRight]);
    //left.w = 0;
    //right.w = 0;

    d_left[idx] = left;
    d_right[idx] = right;

}

CPUTimer_t XReturn::load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading X Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    gpu_time = InvokeGPUKernelParallel(loadXReturnKernel,numBlocks,blockSize,d_left,d_right,d_in,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaMemcpy, h_left, d_left, sizeof(float4)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_right, d_right, sizeof(float4)*size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}

__global__ void storeXReturnKernel(const float4* __restrict d_left, const float4* __restrict d_right, float4* __restrict d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,overload,local_grid_size.y,local_grid_size.z);

    idx3dLeft.y += overload;
    idx3dLeft.z += overload;

    int3 idx3dRight = make_int3((local_grid_size.x - idx3dLeft.x) - 1,idx3dLeft.y,idx3dLeft.z);

    idx3dLeft.x += overload;
    idx3dRight.x += overload;

    //LEFT
    //idx3dLeft is in the local coordinates of the original rank
    //so offset it by overload (now should be negative)
    //transform into our local coordinates
    idx3dLeft.x -= 2*overload;
    idx3dLeft.x += local_grid_size.x + overload + overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    float4 left = __ldg(&d_left[idx]);
    d_out[idxLeft] = left;

    //RIGHT
    //idx3dRight is in the local coordinates of the original rank
    //idx3dRight.x -= local_grid_size.x;
    idx3dRight.x += 2*overload;
    idx3dRight.x -= ol_grid_size.x;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;
    float4 right = __ldg(&d_right[idx]);
    d_out[idxRight] = right;

}

CPUTimer_t XReturn::store(float4* h_left, float4* h_right, float4* d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring X Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    cudaCall(cudaMemcpy, d_left, h_left, sizeof(float4)*size, cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy, d_right, h_right, sizeof(float4)*size, cudaMemcpyHostToDevice);

    gpu_time = InvokeGPUKernelParallel(storeXReturnKernel,numBlocks,blockSize,d_left,d_right,d_out,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}




__global__ void loadYReturnKernel(float4* __restrict d_left, float4* __restrict d_right, const float4* __restrict d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,overload,local_grid_size.z);

    //idx3dLeft.y += overload;
    idx3dLeft.z += overload;

    int3 idx3dRight = make_int3(idx3dLeft.x,(local_grid_size.y - idx3dLeft.y) - 1,idx3dLeft.z);

    idx3dLeft.y += overload;
    idx3dRight.y += overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;

    float4 left = __ldg(&d_in[idxLeft]);
    float4 right = __ldg(&d_in[idxRight]);
    //left.w = 0;
    //right.w = 0;

    d_left[idx] = left;
    d_right[idx] = right;

}

CPUTimer_t YReturn::load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Y Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    gpu_time = InvokeGPUKernelParallel(loadYReturnKernel,numBlocks,blockSize,d_left,d_right,d_in,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaMemcpy, h_left, d_left, sizeof(float4)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_right, d_right, sizeof(float4)*size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}


__global__ void storeYReturnKernel(const float4* __restrict d_left, const float4* __restrict d_right, float4* __restrict d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,overload,local_grid_size.z);

    //idx3dLeft.y += overload;
    idx3dLeft.z += overload;

    int3 idx3dRight = make_int3(idx3dLeft.x,(local_grid_size.y - idx3dLeft.y) - 1,idx3dLeft.z);

    idx3dLeft.y += overload;
    idx3dRight.y += overload;

    //LEFT
    //idx3dLeft is in the local coordinates of the original rank
    //so offset it by overload (now should be negative)
    //transform into our local coordinates
    idx3dLeft.y -= 2*overload;
    idx3dLeft.y += local_grid_size.y + overload + overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    float4 left = __ldg(&d_left[idx]);
    d_out[idxLeft] = left;

    //RIGHT
    //idx3dRight is in the local coordinates of the original rank
    //idx3dRight.x -= local_grid_size.x;
    idx3dRight.y += 2*overload;
    idx3dRight.y -= ol_grid_size.y;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;
    float4 right = __ldg(&d_right[idx]);
    d_out[idxRight] = right;

    /*int3 idx3dRight = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,overload,local_grid_size.z);

    idx3dRight.z += overload;

    int3 idx3dLeft = make_int3(idx3dRight.x,idx3dRight.y + overload + local_grid_size.y,idx3dRight.z);

    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;
    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;

    float4 left = __ldg(&d_left[idx]);
    float4 right = __ldg(&d_right[idx]);

    d_out[idxRight] = right;
    d_out[idxLeft] = left;*/

}

CPUTimer_t YReturn::store(float4* h_left, float4* h_right, float4* d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Y Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    cudaCall(cudaMemcpy, d_left, h_left, sizeof(float4)*size, cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy, d_right, h_right, sizeof(float4)*size, cudaMemcpyHostToDevice);

    gpu_time = InvokeGPUKernelParallel(storeYReturnKernel,numBlocks,blockSize,d_left,d_right,d_out,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}


__global__ void loadZReturnKernel(float4* __restrict d_left, float4* __restrict d_right, const float4* __restrict d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,ol_grid_size.x,overload);

    //idx3dLeft.y += overload;
    //idx3dLeft.z += overload;

    int3 idx3dRight = make_int3(idx3dLeft.x,idx3dLeft.y,(local_grid_size.z - idx3dLeft.z) - 1);

    idx3dLeft.z += overload;
    idx3dRight.z += overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;

    float4 left = __ldg(&d_in[idxLeft]);
    float4 right = __ldg(&d_in[idxRight]);
    //left.w = 0;
    //right.w = 0;

    d_left[idx] = left;
    d_right[idx] = right;

}

CPUTimer_t ZReturn::load(float4* h_left, float4* h_right, float4* d_in, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sLoading Z Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    gpu_time = InvokeGPUKernelParallel(loadZReturnKernel,numBlocks,blockSize,d_left,d_right,d_in,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaMemcpy, h_left, d_left, sizeof(float4)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_right, d_right, sizeof(float4)*size, cudaMemcpyDeviceToHost);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}


__global__ void storeZReturnKernel(const float4* __restrict d_left, const float4* __restrict d_right, float4* __restrict d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;

    int3 idx3dLeft = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,ol_grid_size.x,overload);

    //idx3dLeft.y += overload;
    //idx3dLeft.z += overload;

    int3 idx3dRight = make_int3(idx3dLeft.x,idx3dLeft.y,(local_grid_size.z - idx3dLeft.z) - 1);

    idx3dLeft.z += overload;
    idx3dRight.z += overload;

    //LEFT
    //idx3dLeft is in the local coordinates of the original rank
    //so offset it by overload (now should be negative)
    //transform into our local coordinates
    idx3dLeft.z -= 2*overload;
    idx3dLeft.z += local_grid_size.z + overload + overload;

    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;
    float4 left = __ldg(&d_left[idx]);
    d_out[idxLeft] = left;

    //RIGHT
    //idx3dRight is in the local coordinates of the original rank
    //idx3dRight.x -= local_grid_size.x;
    idx3dRight.z += 2*overload;
    idx3dRight.z -= ol_grid_size.z;
    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;
    float4 right = __ldg(&d_right[idx]);
    d_out[idxRight] = right;

    /*int3 idx3dRight = HACCGPM::parallel::get_local_index(idx,ol_grid_size.x,ol_grid_size.y,overload);

    //idx3dRight.z += overload;

    int3 idx3dLeft = make_int3(idx3dRight.x,idx3dRight.y,idx3dRight.z + overload + local_grid_size.z);

    int idxRight = idx3dRight.x * ol_grid_size.y * ol_grid_size.z + idx3dRight.y * ol_grid_size.z + idx3dRight.z;
    int idxLeft = idx3dLeft.x * ol_grid_size.y * ol_grid_size.z + idx3dLeft.y * ol_grid_size.z + idx3dLeft.z;

    float4 left = __ldg(&d_left[idx]);
    float4 right = __ldg(&d_right[idx]);

    d_out[idxRight] = right;
    d_out[idxLeft] = left;*/

}

CPUTimer_t ZReturn::store(float4* h_left, float4* h_right, float4* d_out, int3 local_grid_size, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);
    CPUTimer_t gpu_time;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%sStoring Z Return Buffers\n",indent);
    #endif

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    #ifdef VerboseGEKernels
    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);
    #endif

    float4* d_left; cudaCall(cudaMalloc,&d_left,sizeof(float4)*size);
    float4* d_right; cudaCall(cudaMalloc,&d_right,sizeof(float4)*size);

    cudaCall(cudaMemcpy, d_left, h_left, sizeof(float4)*size, cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy, d_right, h_right, sizeof(float4)*size, cudaMemcpyHostToDevice);

    gpu_time = InvokeGPUKernelParallel(storeZReturnKernel,numBlocks,blockSize,d_left,d_right,d_out,local_grid_size,ol_grid_size,overload,n);

    cudaCall(cudaFree,d_left);
    cudaCall(cudaFree,d_right);

    return gpu_time;

}