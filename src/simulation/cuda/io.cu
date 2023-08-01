#include "haccgpm.hpp"
#include <stdlib.h>
#include <stdio.h>

CPUTimer_t OUTPUT_TIME = 0;
int OUTPUT_CALLS = 0;

void HACCGPM::serial::writeOutput(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, char* fname, int calls){
    HACCGPM::serial::writeOutput(fname,mem.d_pos,mem.d_vel,params.ng,calls);
}

template<class T>
void HACCGPM::serial::writeOutput(char* fname, T* d_pos, T* d_vel, int ng, int calls){

    getIndent(calls);

    printf("%swriteOutput was called\n",indent);

    CPUTimer_t start = CPUTimer();

    int n_particles = ng*ng*ng;
    T* h_pos = (T*)malloc(sizeof(T)*n_particles);
    T* h_vel = (T*)malloc(sizeof(T)*n_particles);

    cudaCall(cudaMemcpy, h_pos, d_pos, sizeof(T)*ng*ng*ng, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_vel, d_vel, sizeof(T)*ng*ng*ng, cudaMemcpyDeviceToHost);

    FILE *fp;
    fp = fopen(fname, "w+");
    fwrite(h_pos,sizeof(T),n_particles,fp);
    fwrite(h_vel,sizeof(T),n_particles,fp);
    fclose(fp);
    free(h_pos);
    free(h_vel);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;

    printf("%s   writeOutput took %llu us\n",indent,t);

    OUTPUT_CALLS++;
    OUTPUT_TIME += t;

}

template void HACCGPM::serial::writeOutput<float4>(char*,float4*,float4*,int,int);
template void HACCGPM::serial::writeOutput<float3>(char*,float3*,float3*,int,int);

void HACCGPM::serial::printOutputTimes(){
    printf("   writeOutput        -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10d us | mean: %10.2f us\n",OUTPUT_CALLS,OUTPUT_TIME,OUTPUT_TIME,0,((float)OUTPUT_TIME)/((float)(OUTPUT_CALLS)));
}