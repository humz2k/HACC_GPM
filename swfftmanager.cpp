#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include "swfft-all-to-all/include/swfft.hpp"
#include <cstring>

#define VerboseFFT

class FFTManager{
    public:
        Distribution dist;
        Dfft dfft;
        deviceFFT_t* d_scratch;
        FFTManager(HACCGPM::Params& params) : dist(MPI_COMM_WORLD,params.ng,params.blockSize), dfft(dist) {
            params.nlocal = dist.nlocal;
            params.world_rank = dist.world_rank;
            params.world_size = dist.world_size;
            if (params.world_rank == 0)printf("FFTManager:\n   Initialized SWFFT\n");
            cudaCall(cudaMalloc,&d_scratch,sizeof(deviceFFT_t)*params.nlocal);
            if (params.world_rank == 0)printf("   Allocated d_scratch: %lu bytes\n",sizeof(deviceFFT_t)*params.nlocal);
            dfft.makePlans(d_scratch);
            if (params.world_rank == 0)printf("   Made plans\n");
            params.local_grid_start[0] = dist.local_coordinates_start[0];
            params.local_grid_start[1] = dist.local_coordinates_start[1];
            params.local_grid_start[2] = dist.local_coordinates_start[2];
            params.local_grid_size[0] = dist.local_grid_size[0];
            params.local_grid_size[1] = dist.local_grid_size[1];
            params.local_grid_size[2] = dist.local_grid_size[2];
            params.grid_dims[0] = dist.dims[0];
            params.grid_dims[1] = dist.dims[1];
            params.grid_dims[2] = dist.dims[2];
            params.n_particles = params.frac * params.nlocal;
            params.grid_coords[0] = dist.coords[0];
            params.grid_coords[1] = dist.coords[1];
            params.grid_coords[2] = dist.coords[2];
            //printf("world_rank %d, world_size %d\n",params.world_rank,params.world_size);
        }
        ~FFTManager(){
            if (dist.world_rank == 0)printf("FFTManager:\n   Freeing d_scratch...\n");
            cudaCall(cudaFree,d_scratch);
        }
};

FFTManager* ffts;

void HACCGPM::parallel::init_swfft(HACCGPM::Params& params){
    //FFTManager tmp(params);
    ffts = new FFTManager(params);
}

void HACCGPM::parallel::finalize_swfft(){
    delete ffts;
}

void HACCGPM::parallel::forward_fft(deviceFFT_t* d_grid, int ng, int calls){
    getIndent(calls);
    #ifdef VerboseFFT
    if(ffts->dist.world_rank == 0)printf("%sforward_fft was called with\n%s   ng %d\n",indent,indent,ng);
    #endif
    CPUTimer_t start = CPUTimer();
    ffts->dfft.forward(d_grid);
    CPUTimer_t end = CPUTimer();
    if(ffts->dist.world_rank == 0)printf("%s   forward_fft took %llu us\n",indent,end - start);
}

void HACCGPM::parallel::backward_fft(deviceFFT_t* d_grid, int ng, int calls){
    getIndent(calls);
    #ifdef VerboseFFT
    if(ffts->dist.world_rank == 0)printf("%sbackward_fft was called with\n%s   ng %d\n",indent,indent,ng);
    #endif
    CPUTimer_t start = CPUTimer();
    ffts->dfft.backward(d_grid);
    CPUTimer_t end = CPUTimer();
    if(ffts->dist.world_rank == 0)printf("%s   backward_fft took %llu us\n",indent,end - start);
}


