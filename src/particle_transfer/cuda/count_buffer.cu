#include "../kernels.hpp"

__global__ void init_count_buffer(int* ns, const float4* __restrict d_pos, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims){
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

    atomicAdd(&ns[dest_rank],1);
}

CPUTimer_t launch_count_buffer(int* d_ns, float4* d_pos, int n_particles, int ng, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(init_count_buffer,numBlocks,blockSize,d_ns,d_pos,n_particles,ng,local_grid_size,local_coords,dims);
}