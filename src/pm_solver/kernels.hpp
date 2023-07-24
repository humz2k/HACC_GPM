
#include "haccgpm.hpp"

CPUTimer_t launch_getgreens(hostFFT_t* __restrict d_greens, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_getgreens(hostFFT_t* __restrict d_greens, int ng, int nlocal, int3 local_grid_size_vec, int3 grid_coords_vec, int world_rank, int numBlocks, int blockSize,  int calls);

CPUTimer_t launch_combine(float4* d_grad, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_combine(float4* d_grad, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int3 local_grid_size, int3 local_coords, int overload, int nlocal, int ng, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_kspace_solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_kspace_solve_gradient(deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls);