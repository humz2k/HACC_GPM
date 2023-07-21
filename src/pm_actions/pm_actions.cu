#include <stdlib.h>
#include <stdio.h>
#include "pm_kernels.hpp"

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

void HACCGPM::parallel::CIC(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager&mem, int calls){
    HACCGPM::parallel::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.n_particles,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec,params.blockSize,params.world_rank,params.world_size,params.overload,calls);
}

void HACCGPM::parallel::CIC(deviceFFT_t* d_grid, 
                                    float* d_tempgrid, 
                                    float4* d_pos, 
                                    int ng, 
                                    int n_particles, 
                                    int3 local_grid_size,
                                    int3 local_coords,
                                    int3 dims,
                                    int blockSize, 
                                    int world_rank, 
                                    int world_size, 
                                    int overload, 
                                    int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (n_particles + (blockSize - 1))/blockSize;
    //int3 local_grid_size = make_int3(local_grid_size_[0],local_grid_size_[1],local_grid_size_[2]);
    //int3 local_coords = make_int3(local_coords_[0],local_coords_[1],local_coords_[2]);
    //int3 dims = make_int3(dims_[0],dims_[1],dims_[2]);
    getIndent(calls);
    #ifdef VerboseUpdate
    if (world_rank == 0)printf("%sCIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    cudaCall(cudaMemset,d_tempgrid,0,sizeof(float)*(local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload));
    CIC_KERNEL_TIME += InvokeGPUKernelParallel(CICKernelParallel,numBlocks,blockSize,d_tempgrid,d_pos,ng,overload,local_grid_size,n_particles,1.0f);

    HACCGPM::parallel::GridExchange gexch(local_coords,local_grid_size,dims,ng,world_size,world_rank,overload,blockSize);

    gexch.resolve(d_tempgrid,calls+1);
    
    numBlocks = ((local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload) + (blockSize - 1))/blockSize;
    InvokeGPUKernelParallel(float2complex,numBlocks,blockSize,d_grid,d_tempgrid,local_grid_size,overload);

    //deviceFFT_t* h_tempgrid = (deviceFFT_t*)malloc(sizeof(deviceFFT_t)*(local_grid_size.x)*(local_grid_size.y)*(local_grid_size.z));
    //cudaCall(cudaMemcpy, h_tempgrid, d_grid, sizeof(deviceFFT_t)*(local_grid_size.x)*(local_grid_size.y)*(local_grid_size.z), cudaMemcpyDeviceToHost);

    //free(h_tempgrid);
    
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    if (world_rank == 0)printf("%s   CIC took %llu us\n",indent,t);
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::parallel::UpdatePositions(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls){
    HACCGPM::parallel::UpdatePositions(mem.d_pos,mem.d_vel,ts,frac,params.ng,params.n_particles,params.blockSize,params.world_rank,calls);
}

void HACCGPM::parallel::UpdatePositions(float4* d_pos, float4* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int n_particles, int blockSize, int world_rank, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (n_particles + (blockSize - 1))/blockSize;
    float prefactor = ((ts.deltaT)/(ts.aa * ts.aa * ts.adot)) * frac;
    getIndent(calls);
    #ifdef VerboseUpdate
    if(world_rank == 0)printf("%sUpdate Positions was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   frac %g\n%s   prefactor %g\n",indent,indent,blockSize,indent,numBlocks,indent,frac,indent,prefactor);
    #endif
    UPDATE_POS_KERNEL_TIME += InvokeGPUKernelParallel(UpdatePosKernelParallel,numBlocks,blockSize,d_pos,d_vel,prefactor,n_particles);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    if (world_rank == 0)printf("%s   UpdatePositions took %llu us\n",indent,t);
    UPDATE_POS_TIME += t;
    UPDATE_POS_CALLS += 1;
}

void HACCGPM::parallel::UpdateVelocities(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, HACCGPM::Timestepper ts, int calls){
    HACCGPM::parallel::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.n_particles,params.ng,params.overload,params.local_grid_size_vec,params.blockSize,params.world_rank,calls);
}

void HACCGPM::parallel::UpdateVelocities(float4* d_vel, float4* d_grad, float4* d_pos, HACCGPM::Timestepper ts, int n_particles, int ng, int overload, int3 local_grid_size, int blockSize, int world_rank, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (n_particles + (blockSize - 1))/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    if(world_rank == 0)printf("%sUpdate Velocities was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    UPDATE_VEL_KERNEL_TIME += InvokeGPUKernelParallel(ICICKernelParallel,numBlocks,blockSize,d_vel,d_grad,d_pos,ts.deltaT,ts.fscal,overload,local_grid_size,ng,n_particles);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    if(world_rank == 0)printf("%s   UpdateVelocities took %llu us\n",indent,t);
    #endif
    UPDATE_VEL_TIME += t;
    UPDATE_VEL_CALLS += 1;
}

void HACCGPM::parallel::printCICTimes(int world_rank){
    //MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean;
    HACCGPM::parallel::timing_stats(CIC_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(CIC_KERNEL_TIME,&gpu_min,&gpu_max,&gpu_mean);
    //HACCGPM::parallel::timing_stats(TRANSFER_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    if (world_rank != 0)return;
    printf("   CIC                   -> calls: %d\n",CIC_CALLS);
    printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
    printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",(total_mean-gpu_mean),(total_max - gpu_max), (total_min - gpu_min));
    printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
    //printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
    printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / CIC_CALLS,total_max / CIC_CALLS,total_min / CIC_CALLS);
}

void HACCGPM::serial::CIC(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls){
    HACCGPM::serial::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.blockSize,calls);
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

void HACCGPM::serial::UpdateVelocities(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, int calls){
    HACCGPM::serial::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.ng,params.blockSize,calls);
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

void HACCGPM::serial::UpdatePositions(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls){
    HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,frac,params.ng,params.blockSize,calls);
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