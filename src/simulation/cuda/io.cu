#include "haccgpm.hpp"
#include <stdlib.h>
#include <stdio.h>

CPUTimer_t OUTPUT_TIME = 0;
int OUTPUT_CALLS = 0;

void HACCGPM::serial::writeOutput(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, char* fname, int calls){
    HACCGPM::serial::writeOutput(fname,mem.d_pos,mem.d_vel,params.ng,calls);
}

void HACCGPM::serial::writeOutput(char* fname, float4* d_pos, float4* d_vel, int ng, int calls){

    getIndent(calls);

    printf("%swriteOutput was called\n",indent);

    CPUTimer_t start = CPUTimer();

    int n_particles = ng*ng*ng;
    float* h_pos = (float*)malloc(sizeof(float)*4*n_particles);
    float* h_vel = (float*)malloc(sizeof(float)*4*n_particles);

    cudaCall(cudaMemcpy, h_pos, d_pos, sizeof(float)*ng*ng*ng*4, cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy, h_vel, d_vel, sizeof(float)*ng*ng*ng*4, cudaMemcpyDeviceToHost);

    FILE *fp;
    fp = fopen(fname, "w+");
    fwrite(h_pos,sizeof(float),4*n_particles,fp);
    fwrite(h_vel,sizeof(float),4*n_particles,fp);
    fclose(fp);
    free(h_pos);
    free(h_vel);

    CPUTimer_t end = CPUTimer();
    CPUTimer_t t = end-start;

    printf("%s   writeOutput took %llu us\n",indent,t);

    OUTPUT_CALLS++;
    OUTPUT_TIME += t;

}

void HACCGPM::serial::printOutputTimes(){
    printf("   writeOutput        -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10d us | mean: %10.2f us\n",OUTPUT_CALLS,OUTPUT_TIME,OUTPUT_TIME,0,((float)OUTPUT_TIME)/((float)(OUTPUT_CALLS)));
}