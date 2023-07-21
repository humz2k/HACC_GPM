#include "haccgpm.hpp"
#ifndef NOPYTHON
void HACCGPM::serial::PyAnalysis(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, HACCGPM::Timestepper& ts, PyCosmoTools& pytools, int step){
    if (params.do_analysis){
        if (params.analysis[step]){
            printf("Doing Python Analysis Step\n");
            float4* particles = (float4*)malloc(sizeof(float4)*params.ng*params.ng*params.ng);
            float4* vels = (float4*)malloc(sizeof(float4)*params.ng*params.ng*params.ng);
            cudaCall(cudaMemcpy, particles, mem.d_pos, sizeof(float4)*params.ng*params.ng*params.ng, cudaMemcpyDeviceToHost);
            cudaCall(cudaMemcpy, vels, mem.d_vel, sizeof(float4)*params.ng*params.ng*params.ng, cudaMemcpyDeviceToHost);
            float* combined = (float*)malloc(sizeof(float)*params.ng*params.ng*params.ng*7);
            for (int i = 0; i < params.ng*params.ng*params.ng; i++){
                combined[i*7] = particles[i].w;
                combined[i*7 + 1] = particles[i].x;
                combined[i*7 + 2] = particles[i].y;
                combined[i*7 + 3] = particles[i].z;
                combined[i*7 + 4] = vels[i].x;
                combined[i*7 + 5] = vels[i].y;
                combined[i*7 + 6] = vels[i].z;
            }
            free(particles);
            free(vels);
            pytools.analysisStep(combined,params.ng*params.ng*params.ng,7,step,ts.aa,ts.z);
            free(combined);
        }
    }
}
#endif