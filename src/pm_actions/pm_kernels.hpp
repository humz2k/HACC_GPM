#include "haccgpm.hpp"

CPUTimer_t launch_updatepos(float4* d_pos, float4* d_vel, float prefactor, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_updatepos(float3* d_pos, float3* d_vel, float prefactor, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_updatepos(float4* d_pos, float4* d_vel, float prefactor, int n, int3 local_grid_size, int overload, int* do_refresh, int world_rank, int numBlocks, int blockSize, int calls);


template<class T1, class T2>
CPUTimer_t launch_cic(T1* d_grid, T2* d_pos, int ng, float mass, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_cic(float* d_grid, float4* d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_f2c(T* d_out, float* d_in, int ng, int numBlocks, int blockSize, int calls);

//CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int ng, int numBlocks, int blockSize, int calls);
//CPUTimer_t launch_f2c(floatFFT_t* d_out, float* d_in, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_f2c(deviceFFT_t* d_out, float* d_in, int3 local_grid_size, int overload, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_icic(float4* d_vel, float4* d_grad, float4* d_pos, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_icic(float3* d_vel, float4* d_grad, float3* d_pos, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_icic(float4* d_vel, float4* d_grad, float4* d_pos, double deltaT, double fscal, int overload, int3 local_grid_size, int ng, int n_particles, int world_rank, int numBlocks, int blockSize, int calls);