#include <stdlib.h>
#include <stdio.h>
#include "../src/haccgpm.hpp"
#include "gridexchangekernels.hpp"

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

void loadXLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading X Left Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadXLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeXLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring X Left Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeXLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
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

void loadXRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading X Right Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadXRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeXRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring X Right Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeXRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
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

void loadYLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading Y Left Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadYLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeYLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring Y Left Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeYLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
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

void loadYRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading Y Right Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadYRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeYRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring Y Right Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeYRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
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

void loadZLeft(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading Z Left Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadZLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeZLeft(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring Z Left Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeZLeftKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
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

void loadZRight(float* h_out, float* d_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sLoading Z Right Buffer\n",indent);

    int n = size;//ol_grid_size.x * ol_grid_size.y * ol_grid_size.z;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_out; cudaCall(cudaMalloc,&d_out,sizeof(float)*size);
    InvokeGPUKernelParallel(loadZRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    cudaCall(cudaMemcpy, h_out, d_out, sizeof(float)*size, cudaMemcpyDeviceToHost);
    cudaCall(cudaFree,d_out);
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

void storeZRight(float* d_out, float* h_in, int3 ol_grid_size, int overload, int size, int blockSize, int world_rank, int calls){
    getIndent(calls);

    if(world_rank == 0)printf("%sStoring Z Right Buffer\n",indent);

    int n = size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    if(world_rank == 0)printf("%s   blockSize = %d\n",indent,blockSize);
    if(world_rank == 0)printf("%s   numBlocks = %d\n",indent,numBlocks);
    if(world_rank == 0)printf("%s           n = %d\n",indent,n);
    if(world_rank == 0)printf("%s        size = %d\n",indent,size);

    float* d_in; cudaCall(cudaMalloc,&d_in,sizeof(float)*size);
    cudaCall(cudaMemcpy, d_in, h_in, sizeof(float)*size, cudaMemcpyHostToDevice);
    InvokeGPUKernelParallel(storeZRightKernel,numBlocks,blockSize,d_out,d_in,ol_grid_size,overload,n);
    
    cudaCall(cudaFree,d_in);
}