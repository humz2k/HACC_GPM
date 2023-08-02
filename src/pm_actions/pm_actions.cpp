#include <stdlib.h>
#include <stdio.h>
#include "pm_kernels.hpp"
#include <mpi.h>

//#define VerboseUpdate

CPUTimer_t CIC_TIME = 0;
CPUTimer_t CIC_KERNEL_TIME = 0;
CPUTimer_t CIC_MPI_TIME = 0;
int CIC_CALLS = 0;

CPUTimer_t UPDATE_POS_TIME = 0;
CPUTimer_t UPDATE_POS_KERNEL_TIME = 0;
CPUTimer_t UPDATE_POS_MPI_TIME = 0;
int UPDATE_POS_CALLS = 0;

CPUTimer_t UPDATE_VEL_TIME = 0;
CPUTimer_t UPDATE_VEL_KERNEL_TIME = 0;
CPUTimer_t UPDATE_VEL_MPI_TIME = 0;
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
    //cudaCall(cudaMemset,d_tempgrid,0,sizeof(float)*(local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload));
    //CIC_KERNEL_TIME += InvokeGPUKernelParallel(CICKernelParallel,numBlocks,blockSize,d_tempgrid,d_pos,ng,overload,local_grid_size,n_particles,1.0f);

    CIC_KERNEL_TIME += launch_cic(d_tempgrid,d_pos,ng,overload,local_grid_size,n_particles,1.0f,world_rank,numBlocks,blockSize,calls);

    HACCGPM::parallel::GridExchange gexch(local_coords,local_grid_size,dims,ng,world_size,world_rank,overload,blockSize);
    
    CPUTimer_t mpi_start = CPUTimer();

    gexch.resolve(d_tempgrid,calls+1);

    CPUTimer_t mpi_end = CPUTimer();

    CIC_MPI_TIME += mpi_end-mpi_start;
    
    numBlocks = ((local_grid_size.x + 2*overload)*(local_grid_size.y + 2*overload)*(local_grid_size.z + 2*overload) + (blockSize - 1))/blockSize;
    //InvokeGPUKernelParallel(float2complex,numBlocks,blockSize,d_grid,d_tempgrid,local_grid_size,overload);
    CIC_KERNEL_TIME += launch_f2c(d_grid,d_tempgrid,local_grid_size,overload,world_rank,numBlocks,blockSize,calls);

    //deviceFFT_t* h_tempgrid = (deviceFFT_t*)malloc(sizeof(deviceFFT_t)*(local_grid_size.x)*(local_grid_size.y)*(local_grid_size.z));
    //cudaCall(cudaMemcpy, h_tempgrid, d_grid, sizeof(deviceFFT_t)*(local_grid_size.x)*(local_grid_size.y)*(local_grid_size.z), cudaMemcpyDeviceToHost);

    //free(h_tempgrid);
    
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    if (world_rank == 0)printf("%s   CIC took %llu us\n",indent,t);
    #else
    HACCGPM::parallel::printTimingStats("PMAC cicpm",((double)t) * 1e-6);
    #endif
    CIC_TIME += t;
    CIC_CALLS += 1;
}

void HACCGPM::parallel::UpdatePositions(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls){
    if(HACCGPM::parallel::UpdatePositions(mem.d_pos,mem.d_vel,ts,frac,params.ng,params.n_particles,params.local_grid_size_vec,params.overload,params.blockSize,params.world_rank,calls)){
        //if(params.world_rank == 0)printf("DOING TRANSFER!!!!\n");
        HACCGPM::parallel::TransferParticles(params,mem);
    }
}

int HACCGPM::parallel::UpdatePositions(float4* d_pos, float4* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int n_particles, int3 local_grid_size, int overload, int blockSize, int world_rank, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (n_particles + (blockSize - 1))/blockSize;
    float prefactor = ((ts.deltaT)/(ts.aa * ts.aa * ts.adot)) * frac;
    getIndent(calls);
    #ifdef VerboseUpdate
    if(world_rank == 0)printf("%sUpdate Positions was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   frac %g\n%s   prefactor %g\n",indent,indent,blockSize,indent,numBlocks,indent,frac,indent,prefactor);
    #endif
    //UPDATE_POS_KERNEL_TIME += InvokeGPUKernelParallel(UpdatePosKernelParallel,numBlocks,blockSize,d_pos,d_vel,prefactor,n_particles);
    int do_refresh;
    UPDATE_POS_KERNEL_TIME += launch_updatepos(d_pos,d_vel,prefactor,n_particles,local_grid_size,overload,&do_refresh,world_rank,numBlocks,blockSize,calls);
    //printf("DO_REFRESH %d\n",do_refresh);
    int refresh_now;
    CPUTimer_t mpi_start = CPUTimer();
    MPI_Allreduce(&do_refresh,&refresh_now,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
    CPUTimer_t mpi_end = CPUTimer();
    UPDATE_POS_MPI_TIME += mpi_end - mpi_start;

    //printf("refresh_now %d\n",refresh_now);
    //if (refresh_now){
    //    printf("ALLL!!!!\n");
    //}
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    if (world_rank == 0)printf("%s   UpdatePositions took %llu us\n",indent,t);
    #else
    HACCGPM::parallel::printTimingStats("PMAC uppos",((double)t) * 1e-6);
    #endif
    UPDATE_POS_TIME += t;
    UPDATE_POS_CALLS += 1;
    return refresh_now;
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
    UPDATE_VEL_KERNEL_TIME += launch_icic(d_vel,d_grad,d_pos,ts.deltaT,ts.fscal,overload,local_grid_size,ng,n_particles,world_rank,numBlocks,blockSize,calls);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    if(world_rank == 0)printf("%s   UpdateVelocities took %llu us\n",indent,t);
    #else
    HACCGPM::parallel::printTimingStats("PMAC upvel",((double)t) * 1e-6);
    #endif
    UPDATE_VEL_TIME += t;
    UPDATE_VEL_CALLS += 1;
}

void HACCGPM::parallel::printPATimes(int world_rank){
    MPI_Barrier(MPI_COMM_WORLD);
    CPUTimer_t total_min,total_max,total_mean,gpu_min,gpu_max,gpu_mean,mpi_min,mpi_max,mpi_mean,cpu_min,cpu_max,cpu_mean;
    HACCGPM::parallel::timing_stats(CIC_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(CIC_KERNEL_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(CIC_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    HACCGPM::parallel::timing_stats(CIC_TIME - (CIC_KERNEL_TIME + CIC_MPI_TIME),&cpu_min,&cpu_max,&cpu_mean);
    if (world_rank == 0){
        printf("   CIC                   -> calls: %d\n",CIC_CALLS);
        printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
        printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",cpu_mean,cpu_max, cpu_min);
        printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
        printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
        printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / CIC_CALLS,total_max / CIC_CALLS,total_min / CIC_CALLS);
    }
    HACCGPM::parallel::timing_stats(UPDATE_POS_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(UPDATE_POS_KERNEL_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(UPDATE_POS_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    HACCGPM::parallel::timing_stats(UPDATE_POS_TIME - (UPDATE_POS_KERNEL_TIME + UPDATE_POS_MPI_TIME),&cpu_min,&cpu_max,&cpu_mean);
    if (world_rank == 0){
        printf("   UpdatePositions       -> calls: %d\n",UPDATE_POS_CALLS);
        printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
        printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",cpu_mean,cpu_max, cpu_min);
        printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
        printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
        printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / UPDATE_POS_CALLS,total_max / UPDATE_POS_CALLS,total_min / UPDATE_POS_CALLS);
    }
    HACCGPM::parallel::timing_stats(UPDATE_VEL_TIME,&total_min,&total_max,&total_mean);
    HACCGPM::parallel::timing_stats(UPDATE_VEL_KERNEL_TIME,&gpu_min,&gpu_max,&gpu_mean);
    HACCGPM::parallel::timing_stats(UPDATE_VEL_MPI_TIME,&mpi_min,&mpi_max,&mpi_mean);
    HACCGPM::parallel::timing_stats(UPDATE_VEL_TIME - (UPDATE_VEL_KERNEL_TIME + UPDATE_VEL_MPI_TIME),&cpu_min,&cpu_max,&cpu_mean);
    if (world_rank == 0){
        printf("   UpdateVelocities      -> calls: %d\n",UPDATE_VEL_CALLS);
        printf("                               total: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean,total_max,total_min);
        printf("                                 cpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",cpu_mean,cpu_max, cpu_min);
        printf("                                 gpu: %10llu us mean | %10llu us max  | %10llu us min  |\n",gpu_mean,gpu_max,gpu_min);
        printf("                                 mpi: %10llu us mean | %10llu us max  | %10llu us min  |\n",mpi_mean,mpi_max,mpi_min);
        printf("                                 avg: %10llu us mean | %10llu us max  | %10llu us min  |\n",total_mean / UPDATE_VEL_CALLS,total_max / UPDATE_VEL_CALLS,total_min / UPDATE_VEL_CALLS);
    }
}

template<class T1, class T2>
void HACCGPM::serial::CIC(T1* d_grid, float* d_temp, T2* d_pos, int ng, int np, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (np*np*np + (blockSize - 1))/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sCIC (complex,float) was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    float gpscale = (((float)ng/(float)ng));
    float mass = gpscale * gpscale * gpscale;
    //cudaCall(cudaMemset,d_temp,0,sizeof(float)*ng*ng*ng);
    //CIC_KERNEL_TIME += InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_temp,d_pos,ng,1.0f);
    CIC_KERNEL_TIME += launch_cic(d_temp,d_pos,ng,np,mass,numBlocks,blockSize,calls);
    //CIC_KERNEL_TIME += InvokeGPUKernel(float2complex,numBlocks,blockSize,d_grid,d_temp,ng*ng*ng);
    numBlocks = (ng*ng*ng + (blockSize - 1))/blockSize;
    CIC_KERNEL_TIME += launch_f2c(d_grid,d_temp,ng,numBlocks,blockSize,calls);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    printf("%s   CIC (complex,float) took %llu us\n",indent,t);
    #else
    printf("CIC (complex,float): %llu us\n",t);
    #endif
    CIC_TIME += t;
    CIC_CALLS += 1;
}

template void HACCGPM::serial::CIC<deviceFFT_t,float4>(deviceFFT_t*,float*,float4*,int,int,int,int);
template void HACCGPM::serial::CIC<deviceFFT_t,float3>(deviceFFT_t*,float*,float3*,int,int,int,int);
template void HACCGPM::serial::CIC<floatFFT_t,float4>(floatFFT_t*,float*,float4*,int,int,int,int);
template void HACCGPM::serial::CIC<floatFFT_t,float3>(floatFFT_t*,float*,float3*,int,int,int,int);

template<class T1, class T2>
void HACCGPM::serial::CIC(T1* d_grid, T2* d_pos, int ng, int np, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (np*np*np + (blockSize - 1))/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sCIC (complex) was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    float gpscale = (((float)ng/(float)ng));
    float mass = gpscale * gpscale * gpscale;
    //cudaCall(cudaMemset,d_temp,0,sizeof(float)*ng*ng*ng);
    //CIC_KERNEL_TIME += InvokeGPUKernel(CICKernel,numBlocks,blockSize,d_temp,d_pos,ng,1.0f);
    CIC_KERNEL_TIME += launch_cic(d_grid,d_pos,ng,np,mass,numBlocks,blockSize,calls);
    //CIC_KERNEL_TIME += InvokeGPUKernel(float2complex,numBlocks,blockSize,d_grid,d_temp,ng*ng*ng);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    printf("%s   CIC (complex) took %llu us\n",indent,t);
    #else
    printf("CIC (complex): %llu us\n",t);
    #endif
    CIC_TIME += t;
    CIC_CALLS += 1;
}

template void HACCGPM::serial::CIC<deviceFFT_t,float4>(deviceFFT_t*,float4*,int,int,int,int);
template void HACCGPM::serial::CIC<deviceFFT_t,float3>(deviceFFT_t*,float3*,int,int,int,int);
template void HACCGPM::serial::CIC<floatFFT_t,float4>(floatFFT_t*,float4*,int,int,int,int);
template void HACCGPM::serial::CIC<floatFFT_t,float3>(floatFFT_t*,float3*,int,int,int,int);

void HACCGPM::serial::CIC(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls){
    #ifdef USE_TEMP_GRID
    HACCGPM::serial::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.np,params.blockSize,calls);
    #else
    HACCGPM::serial::CIC(mem.d_grid,mem.d_pos,params.ng,params.np,params.blockSize,calls);
    #endif

}

template<class T>
void HACCGPM::serial::UpdateVelocities(T* d_vel, float4* d_grad, T* d_pos, HACCGPM::Timestepper ts, int ng, int np, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (np*np*np + (blockSize - 1))/blockSize;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sUpdate Velocities was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    #endif
    UPDATE_VEL_KERNEL_TIME += launch_icic(d_vel,d_grad,d_pos,ts.deltaT,ts.fscal,ng,np,numBlocks,blockSize,calls);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    printf("%s   UpdateVelocities took %llu us\n",indent,t);
    #else
    printf("UpdateVelocities: %llu us\n",t);
    #endif
    UPDATE_VEL_TIME += t;
    UPDATE_VEL_CALLS += 1;
}

template void HACCGPM::serial::UpdateVelocities<float4>(float4*,float4*,float4*,HACCGPM::Timestepper,int,int,int,int);
template void HACCGPM::serial::UpdateVelocities<float3>(float3*,float4*,float3*,HACCGPM::Timestepper,int,int,int,int);

void HACCGPM::serial::UpdateVelocities(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, int calls){
    HACCGPM::serial::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.ng,params.np,params.blockSize,calls);
}

template<class T>
void HACCGPM::serial::UpdatePositions(T* d_pos, T* d_vel, HACCGPM::Timestepper ts, float frac, int ng, int np, int blockSize, int calls){
    CPUTimer_t start = CPUTimer();
    int numBlocks = (np*np*np + (blockSize - 1))/blockSize;
    float prefactor = ((ts.deltaT)/(ts.aa * ts.aa * ts.adot)) * frac;
    getIndent(calls);
    #ifdef VerboseUpdate
    printf("%sUpdate Positions was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   frac %g\n%s   prefactor %g\n",indent,indent,blockSize,indent,numBlocks,indent,frac,indent,prefactor);
    #endif
    //UPDATE_POS_KERNEL_TIME += InvokeGPUKernel(UpdatePosKernel,numBlocks,blockSize,d_pos,d_vel,prefactor,(float)ng);
    UPDATE_POS_KERNEL_TIME += launch_updatepos(d_pos,d_vel,prefactor,ng,np,numBlocks,blockSize,calls);
    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;
    #ifdef VerboseUpdate
    printf("%s   UpdatePositions took %llu us\n",indent,t);
    #else
    printf("UpdatePositions: %llu us\n",t);
    #endif
    UPDATE_POS_TIME += t;
    UPDATE_POS_CALLS += 1;
}

template void HACCGPM::serial::UpdatePositions<float4>(float4*,float4*,HACCGPM::Timestepper,float,int,int,int,int);
template void HACCGPM::serial::UpdatePositions<float3>(float3*,float3*,HACCGPM::Timestepper,float,int,int,int,int);

void HACCGPM::serial::UpdatePositions(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper ts, float frac, int calls){
    HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,frac,params.ng,params.np,params.blockSize,calls);
}

void HACCGPM::serial::printCICTimes(){
    printf("   CIC                -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",CIC_CALLS,CIC_TIME,CIC_TIME - CIC_KERNEL_TIME,CIC_KERNEL_TIME,((float)CIC_TIME)/((float)(CIC_CALLS)));
    printf("   UpdatePositions    -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",UPDATE_POS_CALLS,UPDATE_POS_TIME,UPDATE_POS_TIME - UPDATE_POS_KERNEL_TIME,UPDATE_POS_KERNEL_TIME,((float)UPDATE_POS_TIME)/((float)(UPDATE_POS_CALLS)));
    printf("   UpdateVelocities   -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",UPDATE_VEL_CALLS,UPDATE_VEL_TIME,UPDATE_VEL_TIME - UPDATE_VEL_KERNEL_TIME,UPDATE_VEL_KERNEL_TIME,((float)UPDATE_VEL_TIME)/((float)(UPDATE_VEL_CALLS)));
}