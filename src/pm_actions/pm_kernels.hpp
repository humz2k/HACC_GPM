#include "haccgpm.hpp"

__global__ void UpdatePosKernel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, float ng);

__global__ void UpdatePosKernelParallel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, int n);

//__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int n);

//__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int3 local_grid_size, int overload);

//__global__ void CICKernel(float* __restrict grid, const float4* __restrict my_pos, int ng, float mass);

//__global__ void CICKernelParallel(float* __restrict d_grid, const float4* __restrict d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass);

CPUTimer_t launch_cic(float* d_grid, float4* d_pos, int ng, float mass, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_cic(float* d_grid, float4* d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int3 local_grid_size, int overload, int world_rank, int numBlocks, int blockSize, int calls);

__global__ void ICICKernel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int ng);

__global__ void ICICKernelParallel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int overload, int3 local_grid_size, int ng, int n_particles);