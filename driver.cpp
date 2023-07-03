#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include <mpi.h>
#include "swfft-all-to-all/include/swfft.hpp"
#include "cambTools/ccamb.h"

int serial(const char* params_file){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::serial::MemoryManager mem(params);
    HACCGPM::serial::fft_cache_plan(params.ng);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    init_python(0);

    if (params.do_analysis){
        import_analysis_module(params.analysis_dir,params.analysis_py);
    }

    HACCGPM::serial::GenerateDisplacementIC(params_file,&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize);
    
    if (!params.do_analysis){
        finalize_python(0);
    }
    
    HACCGPM::serial::InitGreens(mem.d_greens,params.ng,params.blockSize);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    sprintf(stepstr, "%s.particles.ini", params.prefix);
    HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);

    ts.advanceHalfStep();

    for (int step = 0; step < params.lastStep; step++){

        printf("\n=========\nSTEP %d\n",step);

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        HACCGPM::serial::CIC(mem.d_grid,mem.d_pos,params.ng,params.blockSize);
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        if (params.pks[step]){
            sprintf(stepstr, "%s.pk.%d", params.prefix,step);
            HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);
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
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    sprintf(stepstr, "%s.particles.fin", params.prefix);
    HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);

    if (params.do_analysis){
        finalize_python(0);
    }

    CPUTimer_t end = CPUTimer();

    printf("\n\n=========\nTimers:\n");
    HACCGPM::serial::printCICTimes();
    HACCGPM::serial::printFFTTimes();
    HACCGPM::serial::printPowerTimes();
    HACCGPM::serial::printOutputTimes();
    printf("   Total: %5.2g minutes\n",((double)(end-start)) * 1.66667e-8);
    printf("=========\n\n");

    return 0;
}

int parallel(const char* params_file){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    Distribution dist(MPI_COMM_WORLD,params.ng,params.blockSize);
    Dfft dfft(dist);

    int world_rank = dist.world_rank;
    int nlocal = dist.nlocal;

    printf("world_rank %d, nlocal %d\n",world_rank,nlocal);


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