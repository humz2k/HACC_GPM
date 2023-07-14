#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include <mpi.h>
#include "swfft-all-to-all/include/swfft.hpp"
#include "cambTools/ccamb.h"
#include "bdwgc/include/gc.h"

int serial(const char* params_file){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::serial::MemoryManager mem(params);
    HACCGPM::serial::fft_cache_plan(params.ng);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    init_python(0,0);

    if (params.do_analysis){
        import_analysis_module(params.analysis_dir,params.analysis_py);
    }

    CPUTimer_t start_init = CPUTimer();

    HACCGPM::serial::GenerateDisplacementIC(params_file,&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize);
    
    CPUTimer_t end_init = CPUTimer();
    CPUTimer_t init_time = end_init - start_init;

    if (!params.do_analysis){
        finalize_python(0);
    }
    
    HACCGPM::serial::InitGreens(mem.d_greens,params.ng,params.blockSize);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,mem.d_tempgrid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

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
            HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,mem.d_tempgrid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);
        }

        if (params.dumps[step]){
            sprintf(stepstr, "%s.particles.%d", params.prefix,step);
            HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
        }

        if (params.do_analysis){
            if (params.analysis[step]){
                printf("Doing Python Analysis Step\n");
                float* particles = (float*)malloc(sizeof(float)*params.ng*params.ng*params.ng*4);
                cudaCall(cudaMemcpy, particles, mem.d_pos, sizeof(float)*params.ng*params.ng*params.ng*4, cudaMemcpyDeviceToHost);
                call_analysis(step,ts.z,ts.aa,particles,params.ng*params.ng*params.ng,params.ng,params.rl);
                free(particles);
                printf("   Done Python Analysis Step\n");
                cudaCall(cudaDeviceSynchronize);
            }
        }
    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,mem.d_tempgrid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    if(params.dump_final){
        sprintf(stepstr, "%s.particles.fin", params.prefix);
        HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
    }

    if (params.do_analysis){
        finalize_python(0);
    }

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

    init_python(0,params.world_rank);
    
    CPUTimer_t start_init = CPUTimer();

    HACCGPM::parallel::GenerateDisplacementIC(params_file,&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize,params.world_rank,params.world_size,params.nlocal,params.local_grid_size);
    HACCGPM::parallel::initTransferParticles(params,mem);
    CPUTimer_t end_init = CPUTimer();
    CPUTimer_t init_time = end_init - start_init;

    MPI_Barrier(MPI_COMM_WORLD);

    //printf("NP: %d\n",params.n_particles);
    
    HACCGPM::parallel::CIC(mem.d_grid,mem.d_tempgrid,mem.d_pos,params.ng,params.n_particles,params.local_grid_size,params.blockSize,params.world_rank, params.world_size);
    //HACCGPM::parallel::transferParticles(params,mem);

    finalize_python(0);

    HACCGPM::parallel::finalize_swfft();

    MPI_Barrier(MPI_COMM_WORLD);

    CPUTimer_t end = CPUTimer();

    CPUTimer_t init_mean, init_max, init_min;
    HACCGPM::parallel::timing_stats(init_time,&init_min,&init_max,&init_mean);

    if (params.world_rank == 0)printf("\n\n=========\nMPI Stats:\n");
    HACCGPM::parallel::printTransferBytes(params.world_rank);
    if (params.world_rank == 0)printf("=========\n\n");

    //if (params.world_rank == 0){
    if (params.world_rank == 0)printf("\n\n=========\nTimers:\n");
    HACCGPM::parallel::printTransferTimes(params.world_rank);
    HACCGPM::parallel::printFFTStats(params.world_rank);
    HACCGPM::parallel::printCICTimes(params.world_rank);
    if (params.world_rank == 0)printf("   Initialization: mean %llu us, min %llu us, max %llu us (%5.2g minutes)\n",init_mean,init_min,init_max,((double)(init_mean)) * 1.66667e-8);
    if (params.world_rank == 0)printf("   Total: %5.2g minutes\n",((double)(end-start)) * 1.66667e-8);
    if (params.world_rank == 0)printf("=========\n\n");
    //}


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