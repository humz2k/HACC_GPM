#include "haccgpm.hpp"

__global__ void UpdatePosKernel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, float ng);

__global__ void UpdatePosKernelParallel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, int n);

__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int n);

__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int3 local_grid_size, int overload);

__global__ void CICKernel(float* __restrict grid, const float4* __restrict my_pos, int ng, float mass);

__global__ void CICKernelParallel(float* __restrict d_grid, const float4* __restrict d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass);

__global__ void ICICKernel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int ng);