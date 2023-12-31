//#include "haccgpm.hpp"
//#define VerboseCuda
#include "power_kernels.hpp"
#include <stdlib.h>
#include <stdio.h>

//#define VerbosePower

CPUTimer_t POWER_TIME = 0;
int POWER_CALLS = 0;
CPUTimer_t POWER_KERNEL_TIME = 0;

template<class T>
void writePkOutput(const char* fname, T* binVals, int* binCounts, double* kBins, int nbins, int current_fold){
    char fname2[400];
    sprintf(fname2, "%s.fold.%d", fname, current_fold);
    FILE *fp;
    fp = fopen(fname2, "w+");
    for (int i = 0; i < nbins; i++){
        double out = binVals[i]/((double)binCounts[i]);
        if ((out == out) && (binCounts[i] != 0) && (out != 0)){
            fprintf(fp,"%g,%g,%d\n",kBins[i],out,binCounts[i]);
        }
        binVals[i] = 0;
        binCounts[i] = 0;
    }
    fclose(fp);
}

void HACCGPM::parallel::GetPowerSpectrum(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int nbins, const char* fname, int calls){
    CPUTimer_t start = CPUTimer();
    int world_rank = params.world_rank;
    int numBlocks = (params.nlocal + (params.blockSize - 1))/params.blockSize;
    int particleNumBlocks = (params.n_particles + (params.blockSize - 1))/params.blockSize;

    getIndent(calls);

    #ifdef VerbosePower
    if(params.world_rank == 0)printf("%sGetPowerSpectrum was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    if(params.world_rank == 0)printf("%s   Allocating kBins, binCounts, binVals, d_binCounts, d_binVals\n",indent);
    #endif
    
    double* kBins = (double*)malloc(sizeof(double)*nbins);
    int* binCounts = (int*)malloc(sizeof(int)*(nbins));
    double* binVals = (double*)malloc(sizeof(double)*nbins);

    int* d_binCounts; cudaCall(cudaMalloc,&d_binCounts,sizeof(int)*nbins);
    double* d_binVals; cudaCall(cudaMalloc,&d_binVals,sizeof(double)*nbins);

    float4* d_temppos;
    float4* d_tempvel;

    #ifdef VerbosePower
    if(params.world_rank == 0)printf("%s      Allocated kBins, binCounts, binVals, d_binCounts, d_binVals\n",indent);
    if(params.world_rank == 0)printf("%s   Filling kBins\n",indent);
    #endif

    if (params.pkFolds > 0){
        cudaCall(cudaMalloc,&d_temppos,sizeof(float4)*params.n_particles);
        cudaCall(cudaMalloc,&d_tempvel,sizeof(float4)*params.n_particles);
        InvokeGPUKernelParallel(cpy,particleNumBlocks,params.blockSize,d_temppos,mem.d_pos,params.n_particles);
        InvokeGPUKernelParallel(cpy,particleNumBlocks,params.blockSize,d_tempvel,mem.d_vel,params.n_particles);
    }

    double new_rl = (params.rl/(pow(2,params.pkFolds)));

    double d = (2*M_PI)/(new_rl);
    double d3 = sqrt(3*d*d);
    
    double minK = d;

    double maxK = sqrt(3*((params.ng/2)*(params.ng/2)*d*d));

    double binDelta = (maxK - minK)/((double)nbins);

    for (int i = 0; i < nbins; i++){
        kBins[i] = minK + binDelta * (((float)i) + 0.5f);
    }

    #ifdef VerbosePower
    if(params.world_rank == 0)printf("%s      Filled kBins: minK = %g, maxK = %g, binDelta = %g, nbins = %d\n",indent,minK,maxK,binDelta,nbins);
    #endif

    new_rl = params.rl;

    for (int current_fold = 0; current_fold < params.pkFolds + 1; current_fold++){
        cudaCall(cudaMemset,d_binCounts,0,sizeof(int)*nbins);
        cudaCall(cudaMemset,d_binVals,0,sizeof(double)*nbins);

        new_rl = (params.rl/(pow(2,current_fold)));

        d = (2*M_PI)/(new_rl);

        if (current_fold > 0){
            POWER_KERNEL_TIME += InvokeGPUKernelParallel(foldParticles,particleNumBlocks,params.blockSize,mem.d_pos,(double)params.ng,params.local_grid_size_vec,params.grid_coords_vec);
            #ifdef VerbosePower
            if(params.world_rank == 0)printf("%s   Transfering Particles\n",indent);
            #endif
            HACCGPM::parallel::TransferParticles(params,mem,calls+1);
        }

        #ifdef VerbosePower
        if(params.world_rank == 0)printf("%s   Calling CIC\n",indent);
        #endif

        HACCGPM::parallel::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.n_particles,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec,params.blockSize,params.world_rank, params.world_size,params.overload,calls+1);

        #ifdef VerbosePower
        if(params.world_rank == 0)printf("%s      Called CIC\n",indent);
        if(params.world_rank == 0)printf("%s   Calculating PK\n",indent);
        #endif

        HACCGPM::parallel::forward_fft(mem.d_grid,params.ng,calls+1);

        InvokeGPUKernelParallel(scalePower,numBlocks,params.blockSize,mem.d_grid,(double)params.ng,(double)params.ng,params.rl,params.nlocal);
        InvokeGPUKernelParallel(PkCICFilter,numBlocks,params.blockSize,mem.d_grid,params.ng,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec);
        cudaCall(cudaMemset,d_binCounts,0,sizeof(int)*nbins);
        cudaCall(cudaMemset,d_binVals,0,sizeof(double)*nbins);
        InvokeGPUKernelParallel(BinPower,numBlocks,params.blockSize,mem.d_grid,d_binVals,d_binCounts,minK,binDelta,new_rl,params.ng,params.nlocal,params.world_rank,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec);

        #ifdef VerbosePower
        if(params.world_rank == 0)printf("%s      Calculated PK\n",indent);
        if(params.world_rank == 0)printf("%s   Sending values\n",indent);
        #endif

        cudaCall(cudaMemcpy, binVals, d_binVals, sizeof(double)*nbins, cudaMemcpyDeviceToHost);
        cudaCall(cudaMemcpy, binCounts, d_binCounts, sizeof(int)*nbins, cudaMemcpyDeviceToHost);

        HACCGPM::parallel::sendPower(binCounts,binVals,nbins,params.world_rank,params.world_size,calls+1);

        #ifdef VerbosePower
        if(params.world_rank == 0)printf("%s      Sent values\n",indent);
        #endif

        if (params.world_rank == 0)writePkOutput(fname,binVals,binCounts,kBins,nbins,current_fold);
    }
    
    free(kBins);
    free(binCounts);
    free(binVals);
    cudaCall(cudaFree,d_binCounts);
    cudaCall(cudaFree,d_binVals);

    if(params.pkFolds > 0){
        InvokeGPUKernelParallel(cpy,particleNumBlocks,params.blockSize,mem.d_pos,d_temppos,params.n_particles);
        InvokeGPUKernelParallel(cpy,particleNumBlocks,params.blockSize,mem.d_vel,d_tempvel,params.n_particles);
        cudaCall(cudaFree,d_temppos);
        cudaCall(cudaFree,d_tempvel);
    }
}

void HACCGPM::serial::GetPowerSpectrum(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, const char* fname, int calls){

    CPUTimer_t start = CPUTimer();

    int numBlocks = (params.ng*params.ng*params.ng + (params.blockSize - 1))/params.blockSize;
    getIndent(calls);
    #ifdef VerbosePower
    printf("%sGetPowerSpectrum was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    printf("%s   Allocating d_temp_pos, kBins, binCounts, binVals, d_binCounts, d_binVals\n",indent);
    #endif

    int nbins = params.pk_bins;

    particle_t* d_temp_pos = (particle_t*)mem.d_grad;

    double* kBins = (double*)malloc(sizeof(double)*nbins);
    int* binCounts = (int*)malloc(sizeof(int)*(nbins));
    float* binVals = (float*)malloc(sizeof(float)*nbins);

    numBlocks = (params.np*params.np*params.np + (params.blockSize - 1))/params.blockSize;
    POWER_KERNEL_TIME += InvokeGPUKernel(cpy,numBlocks,params.blockSize,d_temp_pos,mem.d_pos,params.np*params.np*params.np);

    double new_rl = (params.rl/(pow(2,params.pkFolds)));

    double d = (2*M_PI)/(new_rl);
    double d3 = sqrt(3*d*d);
    
    double minK = d;

    double maxK = sqrt(3*((params.ng/2)*(params.ng/2)*d*d));

    double binDelta = (maxK - minK)/((double)nbins);

    for (int i = 0; i < nbins; i++){
        kBins[i] = minK + binDelta * (((float)i) + 0.5f);
    }

    new_rl = params.rl;

    for (int current_fold = 0; current_fold < params.pkFolds + 1; current_fold++){

        cudaCall(cudaMemset,mem.d_binCounts,0,sizeof(int)*nbins);
        cudaCall(cudaMemset,mem.d_binVals,0,sizeof(float)*nbins);

        new_rl = (params.rl/(pow(2,current_fold)));

        d = (2*M_PI)/(new_rl);

        if (current_fold > 0){
            numBlocks = (params.ng*params.ng*params.ng + (params.blockSize - 1))/params.blockSize;
            POWER_KERNEL_TIME += InvokeGPUKernel(foldParticles,numBlocks,params.blockSize,d_temp_pos,(double)params.ng,params.np);
        }

        CPUTimer_t start_extras = CPUTimer();

        #ifdef USE_TEMP_GRID
        HACCGPM::serial::CIC(mem.d_grid,mem.d_tempgrid,d_temp_pos,params.ng,params.np,params.blockSize,calls+1);
        #else
	      HACCGPM::serial::CIC(mem.d_grid,d_temp_pos,params.ng,params.np,params.blockSize,calls+1);
        #endif

        HACCGPM::serial::forward_fft(mem.d_grid,params.ng,calls + 1);

        CPUTimer_t end_extras = CPUTimer();

        POWER_KERNEL_TIME += end_extras - start_extras;

        numBlocks = (params.ng*params.ng*params.ng + (params.blockSize - 1))/params.blockSize;

        POWER_KERNEL_TIME += InvokeGPUKernel(scalePower,numBlocks,params.blockSize,mem.d_grid,(double)params.np,(double)params.ng,params.rl,params.ng*params.ng*params.ng);

        POWER_KERNEL_TIME += InvokeGPUKernel(PkCICFilter,numBlocks,params.blockSize,mem.d_grid,params.ng);

        POWER_KERNEL_TIME += InvokeGPUKernel(BinPower,numBlocks,params.blockSize,mem.d_grid,mem.d_binVals,mem.d_binCounts,minK,binDelta,new_rl,params.ng);

        cudaCall(cudaMemcpy, binVals, mem.d_binVals, sizeof(float)*nbins, cudaMemcpyDeviceToHost);
        cudaCall(cudaMemcpy, binCounts, mem.d_binCounts, sizeof(int)*nbins, cudaMemcpyDeviceToHost);
 
        writePkOutput(fname,binVals,binCounts,kBins,nbins,current_fold);

    }
    free(kBins);
    free(binCounts);
    free(binVals);

    CPUTimer_t end = CPUTimer();

    CPUTimer_t t = end-start;
    POWER_CALLS++;
    POWER_TIME += t;
    #ifdef VerbosePower
    printf("%s   GetPowerSpectrum took %llu us\n",indent,t);
    #endif

}

void HACCGPM::serial::printPowerTimes(){
    printf("   GetPowerSpectrum   -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",POWER_CALLS,POWER_TIME,POWER_TIME - POWER_KERNEL_TIME,POWER_KERNEL_TIME,((float)POWER_TIME)/((float)(POWER_CALLS)));
}