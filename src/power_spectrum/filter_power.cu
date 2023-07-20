#include "power_kernels.hpp"

__forceinline__ __device__ double get_pk_cic_filter(int3 idx3d, int ng){
    double d = ((2*M_PI)/(((double)(ng))));

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    float filt1 = sinf(0.5f * kmodes.x) / (0.5 * kmodes.x);
    filt1 = filt1*filt1;
    filt1 = __frcp_rn(filt1 * filt1);
    if (kmodes.x == 0){
        filt1 = 1.0;
    }

    float filt2 = sinf(0.5f * kmodes.y) / (0.5 * kmodes.y);
    filt2 = filt2*filt2;
    filt2 = __frcp_rn(filt2 * filt2);
    if (kmodes.y == 0){
        filt2 = 1.0;
    }

    float filt3 = sinf(0.5f * kmodes.z) / (0.5 * kmodes.z);
    filt3 = filt3*filt3;
    filt3 = __frcp_rn(filt3 * filt3);
    if (kmodes.z == 0){
        filt3 = 1.0;
    }
    double filter = filt1 * filt2 * filt3;

    return filter;

}

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng, int nlocal, int3 local_grid_size, int3 local_coords){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
    
    double filter = get_pk_cic_filter(idx3d,ng);

    deviceFFT_t my_grid = __ldg(&grid[idx]);
    my_grid.x *= filter;
    grid[idx] = my_grid;
}

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    double filter = get_pk_cic_filter(idx3d,ng);

    deviceFFT_t my_grid = __ldg(&grid[idx]);
    my_grid.x *= filter;
    grid[idx] = my_grid;
    
}