#include "swfft-all-to-all/include/swfft.hpp"
#include "haccgpm.hpp"
#include <stdio.h>
#include <stdlib.h>

HACCGPM::parallel::FFTManager::FFTManager(HACCGPM::Params params){

    //world_rank = params.world_rank;
    //world_size = params.world_size;
    //if(world_rank == 0)printf("INITIALIZING SWFFT\n");
}

HACCGPM::parallel::FFTManager::~FFTManager(){
    //if(world_rank == 0)printf("FINALIZING SWFFT\n");
}