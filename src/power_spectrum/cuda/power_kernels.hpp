#include "haccgpm.hpp"

__global__ void scalePower(deviceFFT_t* __restrict data, double ng, double rl, int nlocal);

__global__ void scalePower(floatFFT_t* __restrict data, double ng, double rl, int nlocal);

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng, int nlocal, int3 local_grid_size, int3 local_coords);

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng);

__global__ void PkCICFilter(floatFFT_t* __restrict grid, int ng);

__global__ void foldParticles(float4* __restrict d_pos, double ng);

__global__ void foldParticles(float3* __restrict d_pos, double ng);

__global__ void foldParticles(float4* __restrict d_pos, double ng, int3 local_grid_size, int3 local_coords);

__global__ void ScaleParticles(float4* __restrict d_pos, double oldNg, double newNg);

__global__ void cpy(float4* __restrict dest, const float4* __restrict source);

__global__ void cpy(float3* __restrict dest, const float3* __restrict source);

__global__ void cpy(float4* __restrict dest, const float4* __restrict source, int n);

__global__ void BinPower(const deviceFFT_t* __restrict d_grid, double* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng);

__global__ void BinPower(const floatFFT_t* __restrict d_grid, double* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng);

__global__ void BinPower(const deviceFFT_t* __restrict d_grid, double* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims);