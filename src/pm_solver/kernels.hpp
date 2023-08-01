
#include "haccgpm.hpp"

template<class T>
CPUTimer_t launch_getgreens(T* d_greens, int ng, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_getgreens(T* d_greens, int ng, int nlocal, int3 local_grid_size_vec, int3 grid_coords_vec, int world_rank, int numBlocks, int blockSize,  int calls);

template<class T>
CPUTimer_t launch_combine(float4* d_grad, T* d_x, T* d_y, T* d_z, int ng, int numBlocks, int blockSize, int calls);


CPUTimer_t launch_combine(float4* d_grad, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int3 local_grid_size, int3 local_coords, int overload, int nlocal, int ng, int world_rank, int numBlocks, int blockSize, int calls);


template<class T1, class T2>
CPUTimer_t launch_kspace_solve(T1* d_rho, T2* d_greens, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_x, T1* d_y, T1* d_z, T1* d_rho, T2* d_greens, int ng, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_x, T1* d_y, T1* d_z, T1* d_rho, T2* d_greens, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords, int world_rank, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_kspace_solve_gradient(T1* d_grid, float4* d_grad, T2* d_greens, int dim, int ng, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_kspace_solve_gradient(T* d_grid, float4* d_grad, int dim, int ng, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_kspace_solve_gradient(T* d_x, T* d_y, T* d_z, T* d_rho, int ng, int numBlocks, int blockSize, int calls);

template<class T>
CPUTimer_t launch_grid2float4(float4* d_grad, T* d_grid, int ng, int numBlocks, int blockSize, int calls);

template<class T1, class T2>
CPUTimer_t launch_get_real_grid(T1* d_grid, T2* new_grid, int dim, int ng, int numBlocks, int blockSize, int calls);

