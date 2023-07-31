#include "haccgpm.hpp"

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls);

void launch_generate_rng(floatFFT_t* d_grid1, int ng, int seed, int numBlocks, int blockSize, int calls);

void launch_generate_rng(deviceFFT_t* d_grid1, int ng, int seed, int nlocal, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(deviceFFT_t* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(floatFFT_t* grid, hostFFT_t* scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_amplitudes(floatFFT_t* grid, hostFFT_t* scale, int nlocal, int numBlocks, int blockSize, int calls);

void launch_scale_fft(deviceFFT_t* data, double scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_fft(deviceFFT_t* data, double scale, int nlocal, int numBlocks, int blockSize, int calls);

void launch_scale_fft(floatFFT_t* data, double scale, int nlocal, int world_rank, int numBlocks, int blockSize, int calls);

void launch_scale_fft(floatFFT_t* data, double scale, int nlocal, int numBlocks, int blockSize, int calls);

void launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int numBlocks, int blockSize, int calls);

void launch_interpolate_pk(HACCGPM::CosmoClass& cosmo, hostFFT_t* d_pkScale, int ng, double rl, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

void launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int calls);

void launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int nlocal, int world_rank, int calls);

void launch_transform_density_field(deviceFFT_t* d_grid, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

void launch_transform_density_field(floatFFT_t* d_grid, floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

void launch_transform_density_field(deviceFFT_t* d_grid, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double rl, double z_ini, int ng, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

void launch_transform_density_field(float2* d_grid, deviceFFT_t* d_out, int dim, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

void launch_transform_density_field(float2* d_grid, floatFFT_t* d_out, int dim, double delta, double rl, double z_ini, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float4* d_pos, float4* d_vel, floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float3* d_pos, float3* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float3* d_pos, float3* d_vel, floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int nlocal, int3 local_grid_size, int world_rank, int numBlocks, int blockSize, int calls);

void launch_copy_grid(floatFFT_t* d_grid, float2* new_grid, int numBlocks, int blockSize, int calls);

void launch_copy_grid(deviceFFT_t* d_grid, float2* new_grid, int numBlocks, int blockSize, int calls);

void launch_get_real_grid(deviceFFT_t* d_grid, float4* new_grid, int dim, int numBlocks, int blockSize, int calls);

void launch_get_real_grid(floatFFT_t* d_grid, float4* new_grid, int dim, int numBlocks, int blockSize, int calls);

void launch_get_real_grid(deviceFFT_t* d_grid, float3* new_grid, int dim, int numBlocks, int blockSize, int calls);

void launch_get_real_grid(floatFFT_t* d_grid, float3* new_grid, int dim, int numBlocks, int blockSize, int calls);


void launch_place_particles(float4* d_pos, float4* d_vel, float4* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float4* d_pos, float4* d_vel, float3* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float3* d_pos, float3* d_vel, float4* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);

void launch_place_particles(float3* d_pos, float3* d_vel, float3* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls);