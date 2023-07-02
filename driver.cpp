#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include "swfft-all-to-all/include/swfft.hpp"

int main(int argc, char** argv){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    if (argc != 2){
        printf("USAGE: main params\n");
        return 1;
    }

    HACCGPM::Params params = HACCGPM::read_params(argv[1]);

    HACCGPM::serial::MemoryManager mem(params);
    HACCGPM::serial::fft_cache_plan(params.ng);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    HACCGPM::serial::GenerateDisplacementIC(argv[1],&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize);
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
    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    sprintf(stepstr, "%s.particles.fin", params.prefix);
    HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);

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