#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include <mpi.h>
#include "../swfft-all-to-all/include/swfft.hpp"
#include "../cambTools/ccamb.h"
#include "../bdwgc/include/gc.h"
#include "../pycosmotools/include/pycosmotools.hpp"

//#define NOPYTHON

int serial(const char* params_file){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::serial::MemoryManager mem(params);
    HACCGPM::serial::fft_cache_plan(params.ng);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    HACCGPM::CosmoClass cosmo(params);

    #ifndef NOPYTHON
    init_python(0,0);

    int _coords[3] = {0,0,0};
    int _localgrid[3] = {params.ng,params.ng,params.ng};
    PyCosmoTools pytools(params.ng,params.rl,params.world_size,params.world_rank,_coords,_localgrid);
    #endif

    CPUTimer_t start_init = CPUTimer();

    HACCGPM::serial::GenerateDisplacementIC(params_file,mem,cosmo,params,ts);
    
    CPUTimer_t end_init = CPUTimer();
    CPUTimer_t init_time = end_init - start_init;
    //return 0;

    #ifndef NOPYTHON
    if (!params.do_analysis){
        finalize_python(0);
    }

    if (params.do_analysis){
        pytools.loadPyCosmoNotPython();
        pytools.import(params.analysis_py);
        pytools.initialize();
    }
    #endif
    
    HACCGPM::serial::InitGreens(mem,params);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);

    if (params.dump_init){
        sprintf(stepstr, "%s.particles.ini", params.prefix);
        HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
    }

    ts.advanceHalfStep();

    for (int step = 0; step < params.lastStep; step++){

        printf("\n=========\nSTEP %d\n",step);

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        HACCGPM::serial::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.blockSize);
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        if (params.pks[step]){
            sprintf(stepstr, "%s.pk.%d", params.prefix,step);
            HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);
        }

        if (params.dumps[step]){
            sprintf(stepstr, "%s.particles.%d", params.prefix,step);
            HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
        }

        #ifndef NOPYTHON
        if (params.do_analysis){
            if (params.analysis[step]){
                printf("Doing Python Analysis Step\n");
                float4* particles = (float4*)malloc(sizeof(float4)*params.ng*params.ng*params.ng);
                float4* vels = (float4*)malloc(sizeof(float4)*params.ng*params.ng*params.ng);
                cudaCall(cudaMemcpy, particles, mem.d_pos, sizeof(float4)*params.ng*params.ng*params.ng, cudaMemcpyDeviceToHost);
                cudaCall(cudaMemcpy, vels, mem.d_vel, sizeof(float4)*params.ng*params.ng*params.ng, cudaMemcpyDeviceToHost);
                float* combined = (float*)malloc(sizeof(float)*params.ng*params.ng*params.ng*7);
                for (int i = 0; i < params.ng*params.ng*params.ng; i++){
                    combined[i*7] = particles[i].w;
                    combined[i*7 + 1] = particles[i].x;
                    combined[i*7 + 2] = particles[i].y;
                    combined[i*7 + 3] = particles[i].z;
                    combined[i*7 + 4] = vels[i].x;
                    combined[i*7 + 5] = vels[i].y;
                    combined[i*7 + 6] = vels[i].z;
                }
                free(particles);
                free(vels);
                pytools.analysisStep(combined,params.ng*params.ng*params.ng,7,step,ts.aa,ts.z);
                free(combined);
            }
        }
        #endif
    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);

    if(params.dump_final){
        sprintf(stepstr, "%s.particles.fin", params.prefix);
        HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
    }

    #ifndef NOPYTHON
    if (params.do_analysis){
        pytools.freePython();
    }
    #endif

    CPUTimer_t end = CPUTimer();

    printf("\n\n=========\nTimers:\n");
    HACCGPM::serial::printCICTimes();
    HACCGPM::serial::printFFTTimes();
    HACCGPM::serial::printPowerTimes();
    HACCGPM::serial::printOutputTimes();
    printf("   Initialization: %llu us (%5.2g minutes)\n",init_time,((double)(init_time)) * 1.66667e-8);
    printf("   Total: %5.2g minutes\n",((double)(end-start)) * 1.66667e-8);
    printf("=========\n\n");

    return 0;
}

int parallel(const char* params_file){

    GC_INIT();

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::parallel::init_swfft(params);

    if (params.world_rank == 0){
        printf("GLOBAL_GRID_SIZE: [%d %d %d]\n",params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
        printf("LOCAL_GRID_SIZE: [%d %d %d]\n",params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    }

    HACCGPM::parallel::MemoryManager mem(params);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    HACCGPM::CosmoClass cosmo(params);

    #ifndef NOPYTHON
    init_python(0,params.world_rank);
    #endif
    
    CPUTimer_t start_init = CPUTimer();

    HACCGPM::parallel::GenerateDisplacementIC(params_file, mem, cosmo, params, ts);
    MPI_Barrier(MPI_COMM_WORLD);
    HACCGPM::parallel::TransferParticles(params,mem);
    //HACCGPM::parallel::TransferParticles(params,mem);
    CPUTimer_t end_init = CPUTimer();
    CPUTimer_t init_time = end_init - start_init;

    HACCGPM::parallel::InitGreens(mem,params);

    HACCGPM::parallel::GetPowerSpectrum(params,mem,221,"testpk.pk");

    HACCGPM::parallel::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.n_particles,params.blockSize,params.world_rank);

    
    
    //HACCGPM::parallel::GetPowerSpectrum(mem.d_pos,mem.d_grid,mem.d_tempgrid,params.ng,params.rl,params.overload,params.n_particles,params.local_grid_size,params.grid_coords,params.grid_dims,params.nlocal,221,"testpk1.pk",0,params.blockSize,params.world_rank,params.world_size);

    
    
    CPUTimer_t end = CPUTimer();

    CPUTimer_t init_mean, init_max, init_min;
    HACCGPM::parallel::timing_stats(init_time,&init_min,&init_max,&init_mean);

    if (params.world_rank == 0)printf("\n\n=========\nMPI Stats:\n");
    HACCGPM::parallel::printTransferBytes(params.world_rank);
    HACCGPM::parallel::printGridExchangeBytes(params.world_rank);
    if (params.world_rank == 0)printf("=========\n\n");

    if (params.world_rank == 0)printf("\n\n=========\nTimers:\n");
    HACCGPM::parallel::printTransferTimes(params.world_rank);
    HACCGPM::parallel::printGridExchangeTimes(params.world_rank);
    HACCGPM::parallel::printFFTStats(params.world_rank);
    HACCGPM::parallel::printCICTimes(params.world_rank);
    if (params.world_rank == 0)printf("   Initialization: mean %llu us, min %llu us, max %llu us (%5.2g minutes)\n",init_mean,init_min,init_max,((double)(init_mean)) * 1.66667e-8);
    if (params.world_rank == 0)printf("   Total: %5.2g minutes\n",((double)(end-start)) * 1.66667e-8);
    if (params.world_rank == 0)printf("=========\n\n");

    #ifndef NOPYTHON
    finalize_python(0);
    #endif

    HACCGPM::parallel::finalize_swfft();

    return 0;
}

int main(int argc, char** argv){

    MPI_Init(NULL,NULL);

    int world_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    if (argc != 2){
        printf("USAGE: main params\n");
        return 1;
    }
    int out = 1;

    //char static_array[256];
    //setvbuf(stdout, static_array, _IOFBF, sizeof(static_array));

    if (world_size == 1){
        printf("\n=========\nRUNNING IN SERIAL MODE\n=========\n");
        out = serial(argv[1]);
    } else{
        if (world_rank == 0){
            printf("\n=========\nRUNNING IN PARLLEL MODE\n=========\n");
        }
        out = parallel(argv[1]);
    }

    MPI_Finalize();

    return out;
}