#include "haccgpm.hpp"

template<class T>
__global__ void scalePower(T* __restrict data, double ng, double rl, int nlocal);

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng, int nlocal, int3 local_grid_size, int3 local_coords);

template<class T>
__global__ void PkCICFilter(T* __restrict grid, int ng);

template<class T>
__global__ void foldParticles(T* __restrict d_pos, double ng);

__global__ void foldParticles(float4* __restrict d_pos, double ng, int3 local_grid_size, int3 local_coords);

template<class T>
__global__ void cpy(T* __restrict dest, const T* __restrict source, int n);

template<class T1, class T2>
__global__ void BinPower(const T1* __restrict d_grid, T2* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng);

template<class T1, class T2>
__global__ void BinPower(const T1* __restrict d_grid, T2* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims);