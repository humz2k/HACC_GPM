#include "haccgpm.hpp"
#include <curand.h>
#include <curand_kernel.h>

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls);

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int nlocal, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls);

//__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale, int nlocal);

void launch_scale_fft(deviceFFT_t* data, double scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_fft(deviceFFT_t* data, double scale, int nlocal, int numBlocks, int blockSize, int calls);

void interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int numBlocks, int blockSize, int calls);

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng);

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims);

__global__ void transformDensityField(const deviceFFT_t* __restrict oldGrid, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double rl, double a, int ng);

__global__ void transformDensityField(const deviceFFT_t* __restrict oldGrid, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double rl, double a, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims);

__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng);

__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng, int nx, int ny, int nz, int nlocal, int world_rank);