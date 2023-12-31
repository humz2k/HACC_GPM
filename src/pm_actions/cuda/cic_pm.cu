#include "../pm_kernels.hpp"

template<class T>
__global__ void CICKernel(float* __restrict grid, const T* __restrict my_pos, int ng, int np, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= (np*np*np))return;

    T my_particle = __ldg(&my_pos[idx]);
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

                float mul = dx*dy*dz*mass; //* (1.0f/(ng*ng*ng));

                atomicAdd(&grid[indx],mul);
            }
        }
    }
}

template<class T1, class T>
__global__ void CICKernel(T1* __restrict grid, const T* __restrict my_pos, int ng, int np, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= (np*np*np))return;

    T my_particle = __ldg(&my_pos[idx]);
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

                float mul = dx*dy*dz*mass; //* (1.0f/(ng*ng*ng));

                atomicAdd(&grid[indx].x,mul);
            }
        }
    }
}

__global__ void CICKernelParallel(float* __restrict d_grid, const float4* __restrict d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;

    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;
    my_particle.x += (float)overload;
    my_particle.y += (float)overload;
    my_particle.z += (float)overload;

    int3 overload_grid = make_int3(local_grid_size.x + 2*overload, local_grid_size.y + 2*overload, local_grid_size.z + 2*overload);

    int i = my_particle.x;
    int j = my_particle.y;
    int k = my_particle.z;

    float diffx = my_particle.x - ((float)i);
    float diffy = my_particle.y - ((float)j);
    float diffz = my_particle.z - ((float)k);

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

                float mul = dx*dy*dz*mass; //* (1.0f/(ng*ng*ng));

                atomicAdd(&d_grid[indx],mul);

            }
        }
    }
}

template<class T1, class T2>
CPUTimer_t launch_cic(T1* d_grid, T2* d_pos, int ng, int np, float mass, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    cudaCall(cudaMemset,d_grid,0,sizeof(T1)*ng*ng*ng);
    return InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_grid,d_pos,ng,np,mass);
}

template CPUTimer_t launch_cic<float,float4>(float*,float4*,int,int,float,int,int,int);
template CPUTimer_t launch_cic<float,float3>(float*,float3*,int,int,float,int,int,int);
template CPUTimer_t launch_cic<floatFFT_t,float3>(floatFFT_t*,float3*,int,int,float,int,int,int);
template CPUTimer_t launch_cic<floatFFT_t,float4>(floatFFT_t*,float4*,int,int,float,int,int,int);
template CPUTimer_t launch_cic<deviceFFT_t,float4>(deviceFFT_t*,float4*,int,int,float,int,int,int);
template CPUTimer_t launch_cic<deviceFFT_t,float3>(deviceFFT_t*,float3*,int,int,float,int,int,int);

CPUTimer_t launch_cic(float* d_grid, float4* d_pos, int ng, int overload, int3 local_grid_size, int n_particles, float mass, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    cudaCall(cudaMemset,d_grid,0,sizeof(float)*(local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload));
    return InvokeGPUKernelParallel(CICKernelParallel,numBlocks,blockSize,d_grid,d_pos,ng,overload,local_grid_size,n_particles,mass);
}