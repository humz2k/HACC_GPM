#include "../pm_kernels.hpp"

__global__ void ICICKernelParallel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int overload, int3 local_grid_size, int ng, int n_particles){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;
    float4 my_particle = __ldg(&my_pos[idx]);
    if (my_particle.w < -1)return;

    my_particle.x += (float)overload;
    my_particle.y += (float)overload;
    my_particle.z += (float)overload;

    int3 overload_grid = make_int3(local_grid_size.x + 2*overload, local_grid_size.y + 2*overload, local_grid_size.z + 2*overload);

    float3 my_deltaV = make_float3(0.0,0.0,0.0);
    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    for (int x = 0; x < 2; x++){
        for (int y = 0; y < 2; y++){
            for (int z = 0; z < 2; z++){
                int nx = (i + x);
                int ny = (j + y);
                int nz = (k + z);

                if ((nx < 0) || (nx >= overload_grid.x) || (ny < 0) || (ny >= overload_grid.y) || (nz < 0) || (nz >= overload_grid.z))continue;

                int indx = (nx)*(overload_grid.y)*(overload_grid.z) + (ny)*(overload_grid.z) + nz;

                float dx = diffx;
                if (x == 0){
                    dx = 1 - dx;
                }
                float dy = diffy;
                if (y == 0){
                    dy = 1 - dy;
                }
                float dz = diffz;
                if (z == 0){
                    dz = 1 - dz;
                }

                float4 grad = __ldg(&d_grad[indx]);

                float mul = dx*dy*dz * deltaT * (fscal);//* (1.0f/((double)(ng*ng*ng)));// (1.0f/((double)(ng*ng*ng)));// * deltaT * fscal * (1.0f/((double)(ng*ng*ng)));
                my_deltaV.x += mul*grad.x;
                my_deltaV.y += mul*grad.y;
                my_deltaV.z += mul*grad.z;

                //atomicAdd(&grid[indx].x,(double)mul);
            }
        }
    }
    //if (idx < 100){
    //    printf("%g %g %g\n",my_deltaV.x,my_deltaV.y,my_deltaV.z);
    //}

    float4 my_vel = __ldg(&d_vel[idx]);
    my_vel.x += my_deltaV.x;
    my_vel.y += my_deltaV.y;
    my_vel.z += my_deltaV.z;

    d_vel[idx] = my_vel;

}

__global__ void ICICKernel(float4* __restrict d_vel, const float4* __restrict d_grad, const float4* __restrict my_pos, double deltaT, double fscal, int ng){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_particle = __ldg(&my_pos[idx]);
    float3 my_deltaV = make_float3(0.0,0.0,0.0);
    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = (my_particle.x - (float)i);
    float diffy = (my_particle.y - (float)j);
    float diffz = (my_particle.z - (float)k);

    for (int x = 0; x < 2; x++){
        for (int y = 0; y < 2; y++){
            for (int z = 0; z < 2; z++){
                int nx = (i + x)%ng;
                int ny = (j + y)%ng;
                int nz = (k + z)%ng;
                int indx = (nx)*ng*ng + (ny)*ng + nz;

                float dx = diffx;
                if (x == 0){
                    dx = 1 - dx;
                }
                float dy = diffy;
                if (y == 0){
                    dy = 1 - dy;
                }
                float dz = diffz;
                if (z == 0){
                    dz = 1 - dz;
                }

                float4 grad = __ldg(&d_grad[indx]);

                float mul = dx*dy*dz * deltaT * (fscal);//* (1.0f/((double)(ng*ng*ng)));// (1.0f/((double)(ng*ng*ng)));// * deltaT * fscal * (1.0f/((double)(ng*ng*ng)));
                my_deltaV.x += mul*grad.x;
                my_deltaV.y += mul*grad.y;
                my_deltaV.z += mul*grad.z;

                //atomicAdd(&grid[indx].x,(double)mul);
            }
        }
    }

    float4 my_vel = __ldg(&d_vel[idx]);
    my_vel.x += my_deltaV.x;
    my_vel.y += my_deltaV.y;
    my_vel.z += my_deltaV.z;

    d_vel[idx] = my_vel;

}

CPUTimer_t launch_icic(float4* d_vel, float4* d_grad, float4* d_pos, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(ICICKernel,numBlocks,blockSize,d_vel,d_grad,d_pos,deltaT,fscal,ng);
}

CPUTimer_t launch_icic(float4* d_vel, float4* d_grad, float4* d_pos, double deltaT, double fscal, int overload, int3 local_grid_size, int ng, int n_particles, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(ICICKernelParallel,numBlocks,blockSize,d_vel,d_grad,d_pos,deltaT,fscal,overload,local_grid_size,ng,n_particles);
}