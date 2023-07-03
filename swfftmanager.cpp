
#define GPU
#define alltoall
#include "swfft-all-to-all/include/swfft.hpp"
#include "haccgpm.hpp"
#include <stdio.h>
#include <stdlib.h>

class FFTManager{
    public:
        Distribution dist;
        Dfft dfft;
        FFTManager(HACCGPM::Params params) : dist(MPI_COMM_WORLD,params.ng,params.blockSize), dfft(dist) {
            
        } 
};

FFTManager* ffts;

void HACCGPM::parallel::init_swfft(HACCGPM::Params params){
    FFTManager tmp(params);
    ffts = &tmp;
}
