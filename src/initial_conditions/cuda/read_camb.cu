#include "../ic_kernels.hpp"

#ifndef NOPYTHON
#include "../cambTools/ccamb.h"
#endif

CPUTimer_t launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int calls){

    CPUTimer_t start = CPUTimer();

    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*ng*ng*ng);

    get_pk(fname,h_tmp,z,ng,rl,calls+1);

    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*ng*ng*ng, cudaMemcpyHostToDevice);
    
    free(h_tmp);

    CPUTimer_t end = CPUTimer();

    return end - start;

}

CPUTimer_t launch_get_pk(hostFFT_t* d_pkScale, double z, const char* fname, int ng, double rl, int nlocal, int world_rank, int calls){

    CPUTimer_t start = CPUTimer();

    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*nlocal);

    get_pk_parallel(fname,h_tmp,z,ng,rl,nlocal,world_rank,calls+1);

    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*nlocal, cudaMemcpyHostToDevice);

    free(h_tmp);

    CPUTimer_t end = CPUTimer();

    return end - start;

}