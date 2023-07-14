#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

#define VerboseUpdate

CPUTimer_t CIC_TIME = 0;
CPUTimer_t CIC_KERNEL_TIME = 0;
int CIC_CALLS = 0;

CPUTimer_t UPDATE_POS_TIME = 0;
CPUTimer_t UPDATE_POS_KERNEL_TIME = 0;
int UPDATE_POS_CALLS = 0;

CPUTimer_t UPDATE_VEL_TIME = 0;
CPUTimer_t UPDATE_VEL_KERNEL_TIME = 0;
int UPDATE_VEL_CALLS = 0;

__global__ void UpdatePosKernel(float4* __restrict d_pos, const float4* __restrict d_vel, float prefactor, float ng){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_pos = __ldg(&d_pos[idx]);
    float4 my_vel = __ldg(&d_vel[idx]);
    my_pos.x += my_vel.x * prefactor;
    my_pos.y += my_vel.y * prefactor;
    my_pos.z += my_vel.z * prefactor;
    my_pos.x = fmod(my_pos.x + ng,ng);
    my_pos.y = fmod(my_pos.y + ng,ng);
    my_pos.z = fmod(my_pos.z + ng,ng);
    //if ((my_pos.x < 0 || my_pos.x >= ng) || (my_pos.y < 0 || my_pos.y >= ng) || (my_pos.z < 0 || my_pos.z >= ng)){
    //    printf("%g %g %g\n",my_pos.x,my_pos.y,my_pos.z);
    //    printf("FUCK!!!\n");
    //}
    d_pos[idx] = my_pos;
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

__global__ void CICKernel(deviceFFT_t* __restrict grid, const float4* __restrict my_pos, int ng, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_particle = __ldg(&my_pos[idx]);
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

                atomicAdd(&grid[indx].x,(double)mul);
            }
        }
    }

}

__global__ void CICKernel(float* __restrict grid, const float4* __restrict my_pos, int ng, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    float4 my_particle = __ldg(&my_pos[idx]);
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

__global__ void float2complex(deviceFFT_t* __restrict d_out, const float* __restrict d_in, int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx >= n)return;
    float my_grid = __ldg(&d_in[idx]);
    deviceFFT_t out;
    out.x = my_grid;
    out.y = 0;
    d_out[idx] = out;
}

__global__ void CICKernelParallel(float* __restrict d_grid, const float4* __restrict d_pos, int ng, int3 local_grid_size, int n_particles, float mass){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= n_particles)return;

    float4 my_particle = __ldg(&d_pos[idx]);
    if (my_particle.w < -1)return;

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
                if ((nx < 0) || (nx >= local_grid_size.x) || (ny < 0) || (ny >= local_grid_size.y) || (nz < 0) || (nz >= local_grid_size.z))continue;

                int indx = (nx)*(local_grid_size.y)*(local_grid_size.z) + (ny)*(local_grid_size.z) + nz;

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

void HACCGPM::parallel::CIC(deviceFFT_t* d_grid, float* d_tempgrid, float4* d_pos, int ng, int n_particles, int* local_grid_size_, int blockSize, int world_rank, int world_size, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (n_particles + (blockSize - 1))/blockSize;
    int3 local_grid_size = make_int3(local_grid_size_[0],local_grid_size_[1],local_grid_size_[2]);
    getIndent(calls);
    #ifdef VerboseUpdate
    if (world_rank == 0)printf("%sCIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    cudaCall(cudaMemset,d_tempgrid,0,sizeof(float)*(local_grid_size.x)*(local_grid_size.y)*(local_grid_size.z));
    CIC_KERNEL_TIME += InvokeGPUKernelParallel(CICKernelParallel,numBlocks,blockSize,d_tempgrid,d_pos,ng,local_grid_size,n_particles,1.0f);
    numBlocks = (local_grid_size.x*local_grid_size.y*local_grid_size.z + (blockSize - 1))/blockSize;
    InvokeGPUKernelParallel(float2complex,numBlocks,blockSize,d_grid,d_tempgrid,local_grid_size.x*local_grid_size.y*local_grid_size.z);

    //HACCGPM::parallel::gridExchange(d_extragrid,local_grid_size,world_rank,world_size,blockSize);
    
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    if (world_rank == 0)printf("%s   CIC took %llu us\n",indent,t);
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::parallel::printCICTimes(int world_rank){
    //MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean;
    HACCGPM::parallel::timing_stats(CIC_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(CIC_KERNEL_TIME,&gpu_min,&gpu_max,&gpu_mean);
    //HACCGPM::parallel::timing_stats(TRANSFER_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    if (world_rank != 0)return;
    printf("   CIC                -> calls: %d\n",CIC_CALLS);
    printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
    printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",(total_mean-gpu_mean),(total_max - gpu_max), (total_min - gpu_min));
    printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
    //printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
    printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / CIC_CALLS,total_max / CIC_CALLS,total_min / CIC_CALLS);
}

void HACCGPM::serial::CIC(deviceFFT_t* d_grid, float4* d_pos, int ng, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sCIC (complex) was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    cudaCall(cudaMemset,d_grid,0,sizeof(deviceFFT_t)*ng*ng*ng);
    CIC_KERNEL_TIME += InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_grid,d_pos,ng,1.0f);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    printf("%s   CIC (complex) took %llu us\n",indent,t);
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::serial::CIC(float* d_grid, float4* d_pos, int ng, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sCIC (float) was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    cudaCall(cudaMemset,d_grid,0,sizeof(float)*ng*ng*ng);
    CIC_KERNEL_TIME += InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_grid,d_pos,ng,1.0f);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    printf("%s   CIC (float) took %llu us\n",indent,t);
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::serial::CIC(deviceFFT_t* d_grid, float* d_temp, float4* d_pos, int ng, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sCIC (complex,float) was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    cudaCall(cudaMemset,d_temp,0,sizeof(float)*ng*ng*ng);
    CIC_KERNEL_TIME += InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_temp,d_pos,ng,1.0f);
    CIC_KERNEL_TIME += InvokeGPUKernel(float2complex,numBlocks,blockSize,d_grid,d_temp,ng*ng*ng);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    printf("%s   CIC (complex,float) took %llu us\n",indent,t);
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::serial::UpdateVelocities(float4* d_vel, float4* d_grad, float4* d_pos, HACCGPM::Timestepper ts, int ng, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sUpdate Velocities was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    UPDATE_VEL_KERNEL_TIME += InvokeGPUKernel(ICICKernel,numBlocks,blockSize,d_vel,d_grad,d_pos,ts.deltaT,ts.fscal,ng);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    printf("%s   UpdateVelocities took %llu us\n",indent,t);
    UPDATE_VEL_TIME += t;
    UPDATE_VEL_CALLS += 1;
}

void HACCGPM::serial::UpdatePositions(float4* d_pos, float4* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (ng*ng*ng)/blockSize;
    float prefactor = ((ts.deltaT)/(ts.aa * ts.aa * ts.adot)) * frac;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sUpdate Positions was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   frac %g\n%s   prefactor %g\n",indent,indent,blockSize,indent,numBlocks,indent,frac,indent,prefactor);
    #endif
    UPDATE_POS_KERNEL_TIME += InvokeGPUKernel(UpdatePosKernel,numBlocks,blockSize,d_pos,d_vel,prefactor,(float)ng);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    printf("%s   UpdatePositions took %llu us\n",indent,t);
    UPDATE_POS_TIME += t;
    UPDATE_POS_CALLS += 1;
}

void HACCGPM::serial::printCICTimes(){
    printf("   CIC               -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",CIC_CALLS,CIC_TIME,CIC_TIME - CIC_KERNEL_TIME,CIC_KERNEL_TIME,((float)CIC_TIME)/((float)(CIC_CALLS)));
    printf("   UpdatePositions   -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",UPDATE_POS_CALLS,UPDATE_POS_TIME,UPDATE_POS_TIME - UPDATE_POS_KERNEL_TIME,UPDATE_POS_KERNEL_TIME,((float)UPDATE_POS_TIME)/((float)(UPDATE_POS_CALLS)));
    printf("   UpdateVelocities  -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",UPDATE_VEL_CALLS,UPDATE_VEL_TIME,UPDATE_VEL_TIME - UPDATE_VEL_KERNEL_TIME,UPDATE_VEL_KERNEL_TIME,((float)UPDATE_VEL_TIME)/((float)(UPDATE_VEL_CALLS)));
}