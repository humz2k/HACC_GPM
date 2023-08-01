#include "haccgpm.hpp"

template<class T>
CPUTimer_t launch_generate_rng(T* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int nlocal, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_scale_amplitudes(T* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_scale_amplitudes(T* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_scale_fft(T* data, double scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_scale_fft(T* data, double scale, int nlocal, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int calls);

CPUTimer_t launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int nlocal, int world_rank, int calls);

template<class T>
CPUTimer_t launch_transform_density_field(T* d_grid, T* d_x, T* d_y, T* d_z, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_transform_density_field(deviceFFT_t* d_grid, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double rl, double z_ini, int ng, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_transform_density_field(float2* d_grid, T* d_out, int dim, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_place_particles(T1* d_pos, T1* d_vel, T2* d_x, T2* d_y, T2* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_place_particles(T* d_pos, T* d_vel, float4* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int nlocal, int3 local_grid_size, int world_rank, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_copy_grid(T* d_grid, float2* new_grid, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_get_real_grid(T1* d_grid, T2* new_grid, int dim, int numBlocks, int blockSize, int calls);

