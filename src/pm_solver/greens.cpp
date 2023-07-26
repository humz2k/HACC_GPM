#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include "kernels.hpp"

//#define VerboseGreens

void HACCGPM::serial::InitGreens(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls){

    int numBlocks = (params.ng*params.ng*params.ng)/params.blockSize;

    getIndent(calls);

    #ifdef VerboseGreens
    printf("%sInitGreens was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    printf("%s   Calling getGreens...\n",indent);
    #endif

    launch_getgreens(mem.d_greens,params.ng,numBlocks,params.blockSize,calls);

    #ifdef VerboseGreens
    printf("%s      Called getGreens...\n",indent);
    #endif

}

void HACCGPM::parallel::InitGreens(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int calls){

    int world_rank = params.world_rank;
    int numBlocks = (params.nlocal + (params.blockSize - 1))/params.blockSize;

    getIndent(calls);

    #ifdef VerboseGreens
    if(params.world_rank == 0)printf("%sInitGreens was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,params.blockSize,indent,numBlocks);
    if(params.world_rank == 0)printf("%s   Calling getGreens...\n",indent);
    #endif

    launch_getgreens(mem.d_greens,params.ng,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.world_rank,numBlocks,params.blockSize,calls);

    #ifdef VerboseGreens
    if(params.world_rank == 0)printf("%s      Called getGreens...\n",indent);
    #endif

}