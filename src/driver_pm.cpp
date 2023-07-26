#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include <unistd.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include <mpi.h>
#include "../swfft-all-to-all/include/swfft.hpp"
#include "../cambTools/ccamb.h"
//#include "../pycosmotools/include/pycosmotools.hpp"

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

    HACCGPM::serial::GenerateDisplacementIC(mem,cosmo,params,ts);
    
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
    
    HACCGPM::serial::InitGreens(params,mem);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);

    if (params.dump_init){
        sprintf(stepstr, "%s.particles.ini", params.prefix);
        HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
    }

    ts.advanceHalfStep();

    CPUTimer_t timestepper_start = CPUTimer();

    for (int step = 0; step < params.lastStep; step++){

        printf("\n=========\nSTEP %d\n",step);

        HACCGPM::serial::UpdatePositions(params,mem,ts,0.5);

        HACCGPM::serial::CIC(params,mem);
        HACCGPM::serial::SolveGradient(params,mem);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdateVelocities(params,mem,ts);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdatePositions(params,mem,ts,0.5);

        if (params.pks[step]){
            sprintf(stepstr, "%s.pk.%d", params.prefix,step);
            HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);
        }

        if (params.dumps[step]){
            sprintf(stepstr, "%s.particles.%d", params.prefix,step);
            HACCGPM::serial::writeOutput(params,mem,stepstr);
        }

        #ifndef NOPYTHON
        HACCGPM::serial::PyAnalysis(params,mem,ts,pytools,step);
        #endif

        CPUTimer_t timestepper_im = CPUTimer();
        CPUTimer_t current_timestepper_time = timestepper_im - timestepper_start;
        double time_per_step = ((double)current_timestepper_time) / ((double)step + 1);
        int steps_remaining = (params.lastStep - step);
        double time_remaining = time_per_step * steps_remaining;
        printf("   %g minutes remaining\n",time_remaining * 1.66667e-8);
    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(params,mem,221,stepstr);

    if(params.dump_final){
        sprintf(stepstr, "%s.particles.fin", params.prefix);
        HACCGPM::serial::writeOutput(params,mem,stepstr);
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

    HACCGPM::parallel::GenerateDisplacementIC(mem, cosmo, params, ts);
    MPI_Barrier(MPI_COMM_WORLD);
    HACCGPM::parallel::TransferParticles(params,mem);
    CPUTimer_t end_init = CPUTimer();
    CPUTimer_t init_time = end_init - start_init;

    HACCGPM::parallel::InitGreens(params,mem);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::parallel::GetPowerSpectrum(params,mem,221,stepstr);

    ts.advanceHalfStep();

    CPUTimer_t timestepper_start = CPUTimer();

    for (int step = 0; step < params.lastStep; step++){

        if(params.world_rank == 0)printf("\n=========\nSTEP %d\n",step);

        HACCGPM::parallel::UpdatePositions(params,mem,ts,0.5);

        HACCGPM::parallel::CIC(params,mem);
        HACCGPM::parallel::SolveGradient(params,mem);

        ts.advanceHalfStep();

        HACCGPM::parallel::UpdateVelocities(params,mem,ts);

        ts.advanceHalfStep();

        HACCGPM::parallel::UpdatePositions(params,mem,ts,0.5);

        if (params.pks[step]){
            sprintf(stepstr, "%s.pk.%d", params.prefix,step);
            HACCGPM::parallel::GetPowerSpectrum(params,mem,221,stepstr);
        }
        
        //HACCGPM::parallel::TransferParticles(params,mem);

        CPUTimer_t timestepper_im = CPUTimer();
        CPUTimer_t current_timestepper_time = timestepper_im - timestepper_start;
        double time_per_step = ((double)current_timestepper_time) / ((double)step + 1);
        int steps_remaining = (params.lastStep - step);
        double time_remaining = time_per_step * steps_remaining;
        if(params.world_rank == 0)printf("   %g minutes remaining\n",time_remaining * 1.66667e-8);

    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::parallel::GetPowerSpectrum(params,mem,221,stepstr);
    
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

    if ((argc < 2) && (argc > 3)){
        printf("USAGE: main <params> [+nXX]\n");
        return 1;
    }
    int out = 1;

    char* params_file;
    char* n_proc_flag;
    for (int i = 1; i < argc; i++){
        if (argv[i][0] == '+'){
            n_proc_flag = argv[i];
        } else{
            params_file = argv[i];
        }
    }

    //printf("Params: %s\n N_PROC %s\n",params_file,n_proc_flag);

    //char static_array[256];
    //setvbuf(stdout, static_array, _IOFBF, sizeof(static_array));

    if (world_size == 1){
        if ((argc == 3) && (n_proc_flag != NULL)){
            if ((n_proc_flag[0] == '+') && (n_proc_flag[1] == 'n')){
                int n_proc = atoi(&n_proc_flag[2]);
                if (n_proc != 1){
                    char* args[5];
                    char mpirun[] = "mpirun";
                    char np[] = "-np";
                    char nranks[20];
                    sprintf(nranks,"%d",n_proc);
                    args[0] = mpirun;
                    args[1] = np;
                    args[2] = nranks;
                    args[3] = argv[0];
                    args[4] = params_file;
                    //printf("%s %s %s %s %s\n",args[0],args[1],args[2],args[3],args[4]);
                    execvp("mpirun",args);
                    return 0;
                }
            }
        }
        printf("\n=========\nRUNNING IN SERIAL MODE\n=========\n");
        out = serial(argv[1]);
    } else{
        if (world_rank == 0){
            printf("\n=========\nRUNNING IN PARLLEL MODE\n=========\n");
            printf("   n = %d\n",world_size);
        }
        out = parallel(argv[1]);
    }

    MPI_Finalize();

    return out;
}