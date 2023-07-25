#include <stdlib.h>
#include <stdio.h>
#include "ic_kernels.hpp"
#include <math.h>

//#define NOPYTHON

#ifndef NOPYTHON
#include "../cambTools/ccamb.h"
#endif

#define VerboseInitializer

//#define NOPYTHON

void GenerateFourierAmplitudes(HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, deviceFFT_t* d_grid1, hostFFT_t* d_pkScale, double z, int calls){
    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;

    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateFourierAmplitudes was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,params.blockSize,indent,numBlocks,indent,z);
    printf("%s   Allocating h_tmp...\n",indent);
    #endif

    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*params.ng*params.ng*params.ng);

    #ifdef VerboseInitializer
    printf("%s      Allocated h_tmp.\n",indent);
    printf("%s   Calling generate_rng...\n",indent);
    #endif

    launch_generate_rng(d_grid1,params.ng,params.seed,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Called generate_rng.\n",indent);
    printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::serial::forward_fft(d_grid1,params.ng,calls+1);

    #ifdef VerboseInitializer
    printf("%s      Done Forward FFT...\n",indent);
    
    #endif

    //init_python(calls + 1);
    #ifdef NOPYTHON
    #ifdef VerboseInitializer
    printf("%s   Getting Pk from ipk...\n",indent);
    #endif
    /*double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((params.ng/2)*2*M_PI)/params.rl;
    maxK = sqrt(maxK*maxK*maxK);
    printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }

    InvokeGPUKernel(interpolatePowerSpectrum,numBlocks,params.blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,params.rl,params.ng);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);*/

    interpolate_pk(cosmo,d_pkScale,params.ng,params.rl,numBlocks,params.blockSize,calls);

    #else
    #ifdef VerboseInitializer
    printf("%s   Getting Pk from Camb...\n",indent);
    #endif
    get_pk(params.fname,h_tmp,z,params.ng,params.rl,calls+1);

    #ifdef VerboseInitializer
    printf("%s      Got Pk from Camb.\n",indent);
    printf("%s   Copying Pk from host to device...\n",indent);
    #endif
    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*params.ng*params.ng*params.ng, cudaMemcpyHostToDevice);
    
    #ifdef VerboseInitializer
    printf("%s      Copied Pk from host to device.\n",indent);
    #endif

    #endif
    
    
    #ifdef VerboseInitializer
    printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    launch_scale_amplitudes(d_grid1,d_pkScale,params.ng*params.ng*params.ng,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Scaled Amplitudes.\n",indent);
    printf("%s   Freeing h_tmp...\n",indent);
    #endif

    free(h_tmp);

    #ifdef VerboseInitializer
    printf("%s      Freed h_tmp.\n",indent);
    #endif
}

void HACCGPM::serial::GenerateDisplacementIC(HACCGPM::serial::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls){
    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,params.blockSize,indent,numBlocks,indent,params.fname);
    printf("%s   Calling GenerateFourierAmplitudes...\n",indent);
    #endif

    GenerateFourierAmplitudes(cosmo, params, mem.d_grid, mem.d_greens, params.z_ini, calls+1);

    #ifdef VerboseInitializer
    printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(params.z_ini + 1)) - (ts.deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;
    //get_delta_and_dotDelta(params_file, z,velZ,&delta,&dotDelta,calls+1);
    cosmo.get_delta_and_dotDelta(params.z_ini,velZ,&delta,&dotDelta);
    
    printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    //finalize_python(calls+1);

    #ifdef VerboseInitializer
    printf("%s      Called get_delta_and_dotDelta.\n",indent);
    printf("%s   Calling transformDensityField...\n",indent);
    #endif

    InvokeGPUKernel(transformDensityField,numBlocks,params.blockSize,mem.d_grid,mem.d_x,mem.d_y,mem.d_z,delta,params.rl,1/(1+params.z_ini),params.ng);

    #ifdef VerboseInitializer
    printf("%s      Called transformVelocityField.\n",indent);
    printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(mem.d_x,params.ng,calls+1);
    HACCGPM::serial::backward_fft(mem.d_y,params.ng,calls+1);
    HACCGPM::serial::backward_fft(mem.d_z,params.ng,calls+1);

    double scale_by = 1.0f/((double)(params.ng*params.ng*params.ng));
    int scale_n = params.ng*params.ng*params.ng;
    launch_scale_fft(mem.d_x,scale_by,scale_n,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_y,scale_by,scale_n,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_z,scale_by,scale_n,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    printf("%s      Done Backward FFTs.\n",indent);
    printf("%s   Calling placeParticles...\n",indent);
    #endif

    InvokeGPUKernel(placeParticles,numBlocks,params.blockSize,mem.d_pos,mem.d_vel,mem.d_x,mem.d_y,mem.d_z,delta,dotDelta,params.rl,1/(1+params.z_ini),ts.deltaT,ts.fscal,params.ng);

    #ifdef VerboseInitializer
    printf("%s      Called placeParticles.\n",indent);
    #endif
}


void GenerateFourierAmplitudesParallel(HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, deviceFFT_t* d_grid, hostFFT_t* d_pkScale, double z, int calls){
    int numBlocks = (params.nlocal + (params.blockSize - 1))/params.blockSize;

    int world_rank = params.world_rank;

    getIndent(calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%sGenerateFourierAmplitudesParallel was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,params.blockSize,indent,numBlocks,indent,z);
    if(world_rank == 0)printf("%s   Allocating h_tmp...\n",indent);
    #endif

    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*params.nlocal);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Allocated h_tmp.\n",indent);
    if(world_rank == 0)printf("%s   Calling generate_rng...\n",indent);
    #endif
    launch_generate_rng(d_grid,params.ng,params.seed,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Called generate_rng.\n",indent);
    if(world_rank == 0)printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::parallel::forward_fft(d_grid,params.ng,calls+1);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Done Forward FFT...\n",indent);
    #endif
    #ifdef NOPYTHON
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from ipk...\n",indent);
    #endif
    double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((params.ng/2)*2*M_PI)/params.rl;
    maxK = sqrt(maxK*maxK*maxK);
    if(world_rank == 0)printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        if(world_rank == 0)printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }
    
    InvokeGPUKernelParallel(interpolatePowerSpectrum,numBlocks,params.blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,params.rl,params.ng,params.nlocal,params.world_rank,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);
    #else
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from Camb...\n",indent);
    #endif
    get_pk_parallel(params.fname,h_tmp,z,params.ng,params.rl,params.nlocal,params.world_rank,calls+1);
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Got Pk from Camb.\n",indent);
    #endif
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Copying Pk from host to device...\n",indent);
    #endif

    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*params.nlocal, cudaMemcpyHostToDevice);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Copied Pk from host to device.\n",indent);
    if(world_rank == 0)printf("%s   Scaling Amplitudes...\n",indent);
    #endif
    #endif

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    launch_scale_amplitudes(d_grid,d_pkScale,params.nlocal,world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Scaled Amplitudes.\n",indent);
    if(world_rank == 0)printf("%s   Freeing h_tmp...\n",indent);
    #endif

    free(h_tmp);
    //cudaFree(d_pkScale);
    //cudaFree(rngState);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Freed h_tmp.\n",indent);
    #endif
}

void HACCGPM::parallel::GenerateDisplacementIC(HACCGPM::parallel::MemoryManager& mem, HACCGPM::CosmoClass& cosmo, HACCGPM::Params& params, HACCGPM::Timestepper& ts, int calls){

    int world_rank = params.world_rank;

    int numBlocks = (params.nlocal)/params.blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,params.blockSize,indent,numBlocks,indent,params.fname);
    if (params.world_rank == 0)printf("%s   Calling GenerateFourierAmplitudesParallel...\n",indent);
    #endif

    GenerateFourierAmplitudesParallel(cosmo, params, mem.d_grid, mem.d_greens, params.z_ini, calls+1);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(params.z_ini + 1)) - (ts.deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;
    cosmo.get_delta_and_dotDelta(params.z_ini,velZ,&delta,&dotDelta);

    if (params.world_rank == 0)printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called get_delta_and_dotDelta.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling transformDensityField...\n",indent);
    #endif

    InvokeGPUKernelParallel(transformDensityField,numBlocks,params.blockSize,mem.d_grid,mem.d_x,mem.d_y,mem.d_z,delta,params.rl,1/(1+params.z_ini),params.ng,params.nlocal,params.world_rank,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called transformVelocityField.\n",indent);
    if (params.world_rank == 0)printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::parallel::backward_fft(mem.d_x,params.ng,calls+1);
    HACCGPM::parallel::backward_fft(mem.d_y,params.ng,calls+1);
    HACCGPM::parallel::backward_fft(mem.d_z,params.ng,calls+1);

    double scale_by = 1.0f/((double)(params.ng*params.ng*params.ng));

    launch_scale_fft(mem.d_x,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_y,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);
    launch_scale_fft(mem.d_z,scale_by,params.nlocal,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Done Backward FFTs.\n",indent);
    if (params.world_rank == 0)printf("%s   Calling placeParticles.\n",indent);
    #endif

    InvokeGPUKernelParallel(placeParticles,numBlocks,params.blockSize,mem.d_pos,mem.d_vel,mem.d_x,mem.d_y,mem.d_z,delta,dotDelta,params.rl,1/(1+params.z_ini),ts.deltaT,ts.fscal,params.ng,params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2], params.nlocal, params.world_rank);

    #ifdef VerboseInitializer
    if (params.world_rank == 0)printf("%s      Called placeParticles.\n",indent);
    #endif

}