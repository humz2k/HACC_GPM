#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

CPUTimer_t launch_count_buffer(int* d_ns, float4* d_pos, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_load_buffer(float4* d_swap, int* d_starts, int* d_count, float4* d_pos, float4* d_vel, int n_particles, int ng, int3 local_grid_size, int3 local_coords,int3 dims, int world_rank, int numBlocks, int blockSize, int calls);

CPUTimer_t launch_combine_particles(float4* d_pos, float4* d_vel, float4* d_swap, int* d_count, int n_new, int n_particles, int world_rank, int numBlocks, int blockSize, int calls);