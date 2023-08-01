#include "power_kernels.hpp"

template<class T1, class T2>
__forceinline__ __device__ void calc_power_bins(T1* __restrict d_binVals, int* __restrict d_binCounts, T2 this_val, int idx, int3 idx3d, double minK, double binDelta, double rl, int ng){
    if ((idx3d.x == 0) && (idx3d.y == 0) && (idx3d.z == 0))return;
    if ((idx3d.x == ng/2) && (idx3d.y == ng/2) && (idx3d.z == ng/2))return;

    double d = (2*M_PI)/(rl);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    float kbin = sqrtf(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z) - minK;
    int indx = (int)(kbin/binDelta);

    atomicAdd(&d_binVals[indx],this_val);
    atomicAdd(&d_binCounts[indx],1);
}

template<class T1, class T2>
__global__ void BinPower(const T1* __restrict d_grid, T2* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    T1 this_val = __ldg(&d_grid[idx]);

    calc_power_bins(d_binVals,d_binCounts,this_val.x,idx,idx3d,minK,binDelta,rl,ng);

}

template __global__ void BinPower<deviceFFT_t,double>(const deviceFFT_t* __restrict,double* __restrict,int* __restrict, double, double, double, int);
template __global__ void BinPower<deviceFFT_t,float>(const deviceFFT_t* __restrict,float* __restrict,int* __restrict, double, double, double, int);
template __global__ void BinPower<floatFFT_t,double>(const floatFFT_t* __restrict,double* __restrict,int* __restrict, double, double, double, int);
template __global__ void BinPower<floatFFT_t,float>(const floatFFT_t* __restrict,float* __restrict,int* __restrict, double, double, double, int);

template<class T1, class T2>
__global__ void BinPower(const T1* __restrict d_grid, T2* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    T1 this_val = __ldg(&d_grid[idx]);

    calc_power_bins(d_binVals,d_binCounts,this_val.x,idx,idx3d,minK,binDelta,rl,ng);

}

template __global__ void BinPower<deviceFFT_t,double>(const deviceFFT_t* __restrict,double* __restrict,int* __restrict, double, double, double, int,int,int,int3,int3,int3);
template __global__ void BinPower<deviceFFT_t,float>(const deviceFFT_t* __restrict,float* __restrict,int* __restrict, double, double, double, int,int,int,int3,int3,int3);
template __global__ void BinPower<floatFFT_t,double>(const floatFFT_t* __restrict,double* __restrict,int* __restrict, double, double, double, int,int,int,int3,int3,int3);
template __global__ void BinPower<floatFFT_t,float>(const floatFFT_t* __restrict,float* __restrict,int* __restrict, double, double, double, int,int,int,int3,int3,int3);