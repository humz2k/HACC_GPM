#include <stdio.h>
#include <stdlib.h>
#include "haccgpm.hpp"

#define VerboseFFT

#define FFTCacheSize 10

#define UsePlanManager

CPUTimer_t FFT_FORWARD_TIME_IP = 0;
CPUTimer_t FFT_FORWARD_PLAN_TIME_IP = 0;
CPUTimer_t FFT_BACKWARD_TIME_IP = 0;
CPUTimer_t FFT_BACKWARD_PLAN_TIME_IP = 0;
CPUTimer_t FFT_FORWARD_TIME_OP = 0;
CPUTimer_t FFT_FORWARD_PLAN_TIME_OP = 0;
CPUTimer_t FFT_BACKWARD_TIME_OP = 0;
CPUTimer_t FFT_BACKWARD_PLAN_TIME_OP = 0;

int FFT_FORWARD_CALLS_IP = 0;
int FFT_FORWARD_CALLS_OP = 0;
int FFT_BACKWARD_CALLS_IP = 0;
int FFT_BACKWARD_CALLS_OP = 0;

class PlanManager{
    public:
        cufftHandle plans[FFTCacheSize];
        int ngs[FFTCacheSize];
        int used;
        PlanManager(){
            //printf("FFT PlanManager:\n");
            //printf("   Initialized\n");
            used = 0;
        }
        ~PlanManager(){
            if (used != 0){
                printf("FFT PlanManager:\n");
                for (int i = 0; i < used; i++){
                    printf("   Freeing plan %d\n",ngs[i]);
                    cufftDestroy(plans[i]);
                }
            } else {
                //printf("   No plans to free\n");
            }
        }
        cufftHandle get_plan(int ng, int calls){
            getIndent(calls);
            printf("%sFFT PlanManager:\n",indent);
            printf("%s   Searching for cached plan %d\n",indent,ng);
            for (int i = 0; i < used; i++){
                if (ngs[i] == ng){
                    printf("%s   Found cached plan %d\n",indent,ngs[i]);
                    return plans[i];
                }
            }
            if (used >= FFTCacheSize){
                printf("%s   No more space to create new plan!!!\n",indent);
                exit(1);
            }
            printf("%s   No cached plan %d found, creating one\n",indent,ng);
            if (cufftPlan3d(&plans[used], ng, ng, ng, CUFFT_Z2Z) != CUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            };
            ngs[used] = ng;
            used++;
            return plans[used-1];
        }
};

#ifdef UsePlanManager
PlanManager plan_manager;
#endif

void HACCGPM::serial::fft_cache_plan(int ng){
    #ifdef UsePlanManager
    plan_manager.get_plan(ng,0);
    #endif
}

void HACCGPM::serial::forward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls){

    CPUTimer_t start = CPUTimer();

    getIndent(calls);

    #ifdef VerboseFFT
    printf("%sforward_fft (out of place) was called with\n%s   ng %d\n",indent,indent,ng);
    #endif

    cufftHandle plan;
    #ifdef UsePlanManager
    plan = plan_manager.get_plan(ng,calls+1);
    #else
    printf("%s   Creating plan %d\n",indent,ng);
    if (cufftPlan3d(&plan, ng, ng, ng, CUFFT_Z2Z) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;
    };
    #endif

    CPUTimer_t plan_t = CPUTimer();

    printf("%s   Executing Z2Z CUFFT_FORWARD\n",indent);
    if (cufftExecZ2Z(plan, data, out, CUFFT_FORWARD) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }
    cudaDeviceSynchronize();
    #ifndef UsePlanManager
    printf("%s   Destroying plan %d\n",indent,ng);
    cufftDestroy(plan);
    #endif

    CPUTimer_t end = CPUTimer();
    CPUTimer_t plan_time = plan_t - start;
    CPUTimer_t t = end-start;
    printf("%s   forward_fft (out of place) took %llu us (%llu us planning)\n",indent,t,plan_time);
    FFT_FORWARD_CALLS_OP++;
    FFT_FORWARD_TIME_OP += t;
    FFT_FORWARD_PLAN_TIME_OP += plan_time;
}

void HACCGPM::serial::forward_fft(deviceFFT_t* data, int ng, int calls){

    CPUTimer_t start = CPUTimer();

    getIndent(calls);

    #ifdef VerboseFFT
    printf("%sforward_fft (in place) was called with\n%s   ng %d\n",indent,indent,ng);
    #endif

    cufftHandle plan;
    #ifdef UsePlanManager
    plan = plan_manager.get_plan(ng,calls+1);
    #else
    printf("%s   Creating plan %d\n",indent,ng);
    if (cufftPlan3d(&plan, ng, ng, ng, CUFFT_Z2Z) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;
    };
    #endif
    
    CPUTimer_t plan_t = CPUTimer();

    printf("%s   Executing Z2Z CUFFT_FORWARD\n",indent);
    if (cufftExecZ2Z(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }

    cudaDeviceSynchronize();

    #ifndef UsePlanManager
    printf("%s   Destroying plan %d\n",indent,ng);
    cufftDestroy(plan);
    #endif

    CPUTimer_t end = CPUTimer();
    CPUTimer_t plan_time = plan_t - start;
    CPUTimer_t t = end-start;
    printf("%s   forward_fft (in place) took %llu us (%llu us planning)\n",indent,t,plan_time);
    FFT_FORWARD_CALLS_IP++;
    FFT_FORWARD_TIME_IP += t;
    FFT_FORWARD_PLAN_TIME_IP += plan_time;
}

void HACCGPM::serial::backward_fft(deviceFFT_t* data, deviceFFT_t* out, int ng, int calls){

    CPUTimer_t start = CPUTimer();

    getIndent(calls);

    #ifdef VerboseFFT
    printf("%sbackward_fft (out of place) was called with\n%s   ng %d\n",indent,indent,ng);
    #endif

    cufftHandle plan;
    #ifdef UsePlanManager
    plan = plan_manager.get_plan(ng,calls+1);
    #else
    printf("%s   Creating plan %d\n",indent,ng);
    if (cufftPlan3d(&plan, ng, ng, ng, CUFFT_Z2Z) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;
    };
    #endif

    CPUTimer_t plan_t = CPUTimer();

    printf("%s   Executing Z2Z CUFFT_INVERSE\n",indent);
    if (cufftExecZ2Z(plan, data, out, CUFFT_INVERSE) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Backward failed\n");
        return;	
    }
    cudaDeviceSynchronize();

    #ifndef UsePlanManager
    printf("%s   Destroying plan %d\n",indent,ng);
    cufftDestroy(plan);
    #endif

    CPUTimer_t end = CPUTimer();
    CPUTimer_t plan_time = plan_t - start;
    CPUTimer_t t = end-start;
    printf("%s   backward_fft (out of place) took %llu us (%llu us planning)\n",indent,t,plan_time);
    FFT_BACKWARD_CALLS_OP++;
    FFT_BACKWARD_TIME_OP += t;
    FFT_BACKWARD_PLAN_TIME_OP += plan_time;
}

void HACCGPM::serial::backward_fft(deviceFFT_t* data, int ng, int calls){

    CPUTimer_t start = CPUTimer();

    getIndent(calls);

    #ifdef VerboseFFT
    printf("%sbackward_fft (in place) was called with\n%s   ng %d\n",indent,indent,ng);
    #endif

    cufftHandle plan;
    #ifdef UsePlanManager
    plan = plan_manager.get_plan(ng,calls+1);
    #else
    printf("%s   Creating plan %d\n",indent,ng);
    if (cufftPlan3d(&plan, ng, ng, ng, CUFFT_Z2Z) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;
    };
    #endif
    CPUTimer_t plan_t = CPUTimer();
    printf("%s   Executing Z2Z CUFFT_INVERSE\n",indent);
    if (cufftExecZ2Z(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Backward failed\n");
        return;	
    }
    cudaDeviceSynchronize();
    #ifndef UsePlanManager
    printf("%s   Destroying plan %d\n",indent,ng);
    cufftDestroy(plan);
    #endif
    CPUTimer_t end = CPUTimer();
    CPUTimer_t plan_time = plan_t - start;
    CPUTimer_t t = end-start;
    printf("%s   backward_fft (in place) took %llu us (%llu us planning)\n",indent,t,plan_time);
    FFT_BACKWARD_CALLS_IP++;
    FFT_BACKWARD_TIME_IP += t;
    FFT_BACKWARD_PLAN_TIME_IP += plan_time;
}

void HACCGPM::serial::printFFTTimes(){
    //printf("   CIC              -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",CIC_CALLS,CIC_TIME,CIC_TIME - CIC_KERNEL_TIME,CIC_KERNEL_TIME,((float)CIC_TIME)/((float)(CIC_CALLS)));
    //printf("   UpdatePositions  -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",UPDATE_POS_CALLS,UPDATE_POS_TIME,UPDATE_POS_TIME - UPDATE_POS_KERNEL_TIME,UPDATE_POS_KERNEL_TIME,((float)UPDATE_POS_TIME)/((float)(UPDATE_POS_CALLS)));
    if (FFT_FORWARD_CALLS_OP != 0){
        printf("   forward_fft (op)  -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",FFT_FORWARD_CALLS_OP,FFT_FORWARD_TIME_OP,FFT_FORWARD_PLAN_TIME_OP,FFT_FORWARD_TIME_OP - FFT_FORWARD_PLAN_TIME_OP,((float)FFT_FORWARD_TIME_OP)/((float)(FFT_FORWARD_CALLS_OP)));
    }
    if (FFT_FORWARD_CALLS_IP != 0){
        printf("   forward_fft (ip)  -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",FFT_FORWARD_CALLS_IP,FFT_FORWARD_TIME_IP,FFT_FORWARD_PLAN_TIME_IP,FFT_FORWARD_TIME_IP - FFT_FORWARD_PLAN_TIME_IP,((float)FFT_FORWARD_TIME_IP)/((float)(FFT_FORWARD_CALLS_IP)));
    }
    if (FFT_BACKWARD_CALLS_OP != 0){
        printf("   backward_fft (op) -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",FFT_BACKWARD_CALLS_OP,FFT_BACKWARD_TIME_OP,FFT_BACKWARD_PLAN_TIME_OP,FFT_BACKWARD_TIME_OP - FFT_BACKWARD_PLAN_TIME_OP,((float)FFT_BACKWARD_TIME_OP)/((float)(FFT_BACKWARD_CALLS_OP)));
    }
    if (FFT_BACKWARD_CALLS_IP != 0){
        printf("   backward_fft (ip) -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",FFT_BACKWARD_CALLS_IP,FFT_BACKWARD_TIME_IP,FFT_BACKWARD_PLAN_TIME_IP,FFT_BACKWARD_TIME_IP - FFT_BACKWARD_PLAN_TIME_IP,((float)FFT_BACKWARD_TIME_IP)/((float)(FFT_BACKWARD_CALLS_IP)));
    }
}