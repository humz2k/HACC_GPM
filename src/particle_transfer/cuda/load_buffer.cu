#include "../kernels.hpp"

__global__ void init_load_buffer(float4* d_swap, int* ns, int* counts, float4* __restrict d_pos, float4* __restrict d_vel, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;

    my_particle.x += (float)(local_grid_size.x * local_coords.x);
    my_particle.y += (float)(local_grid_size.y * local_coords.y);
    my_particle.z += (float)(local_grid_size.z * local_coords.z);

    my_particle.x += (float)ng;
    my_particle.y += (float)ng;
    my_particle.z += (float)ng;

    my_particle.x = fmod(my_particle.x,(float)ng);
    my_particle.y = fmod(my_particle.y,(float)ng);
    my_particle.z = fmod(my_particle.z,(float)ng);

    int3 dest_coords = make_int3(my_particle.x / local_grid_size.x,my_particle.y / local_grid_size.y,my_particle.z / local_grid_size.z);

    int dest_rank = dest_coords.x * dims.y * dims.z + dest_coords.y * dims.z + dest_coords.z;

    if (dest_rank == world_rank)return;

    float3 grid_start = make_float3(dest_coords.x * local_grid_size.x, dest_coords.y * local_grid_size.y, dest_coords.z * local_grid_size.z);

    my_particle.x -= grid_start.x;
    my_particle.y -= grid_start.y;
    my_particle.z -= grid_start.z;

    int start = ns[dest_rank];

    int count = atomicAdd(&counts[dest_rank],1);

    int indx = start + count;

    d_swap[indx*2] = my_particle;
    d_swap[(indx*2)+1] = __ldg(&d_vel[idx]);

    d_pos[idx] = make_float4(0,0,0,-10);
    d_vel[idx] = make_float4(0,0,0,-10);

}

CPUTimer_t launch_load_buffer(float4* d_swap, int* d_starts, int* d_count, float4* d_pos, float4* d_vel, int n_particles, int ng, int3 local_grid_size, int3 local_coords,int3 dims, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(init_load_buffer,numBlocks,blockSize,d_swap,d_starts,d_count,d_pos,d_vel,n_particles,ng,local_grid_size,local_coords,dims,world_rank);
}