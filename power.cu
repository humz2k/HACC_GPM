#include "haccgpm.hpp"
#include <stdlib.h>
#include <stdio.h>

#define VerbosePower

CPUTimer_t POWER_TIME = 0;
int POWER_CALLS = 0;
CPUTimer_t POWER_KERNEL_TIME = 0;

__global__ void scalePower(const deviceFFT_t* __restrict data, hostFFT_t* __restrict out, double ng, double rl){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double scale = (1.0f/(ng*ng*ng)) * (rl/ng);
    deviceFFT_t old = __ldg(&data[idx]);
    hostFFT_t absVal = old.x*old.x + old.y*old.y;
    out[idx] = absVal * scale;

}

__global__ void scalePower(deviceFFT_t* __restrict data, double ng, double rl){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double scale = (1.0f/(ng*ng*ng)) * (rl/ng);
    deviceFFT_t old = __ldg(&data[idx]);
    old.x = (old.x * old.x + old.y * old.y) * scale;
    data[idx] = old;

}

__global__ void PkCICFilter(double* __restrict grid, int ng, int folds){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    float filt1 = sinf(0.5f * kmodes.x) / (0.5 * kmodes.x);
    filt1 = filt1*filt1;
    filt1 = __frcp_rn(filt1 * filt1);
    if (kmodes.x == 0){
        filt1 = 1.0;
    }

    float filt2 = sinf(0.5f * kmodes.y) / (0.5 * kmodes.y);
    filt2 = filt2*filt2;
    filt2 = __frcp_rn(filt2 * filt2);
    if (kmodes.y == 0){
        filt2 = 1.0;
    }

    float filt3 = sinf(0.5f * kmodes.z) / (0.5 * kmodes.z);
    filt3 = filt3*filt3;
    filt3 = __frcp_rn(filt3 * filt3);
    if (kmodes.z == 0){
        filt3 = 1.0;
    }
    double filter = filt1 * filt2 * filt3;

    double my_grid = __ldg(&grid[idx]);
    my_grid *= filter;
    grid[idx] = my_grid;
    
}

__global__ void PkCICFilter(deviceFFT_t* __restrict grid, int ng, int folds){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    float filt1 = sinf(0.5f * kmodes.x) / (0.5 * kmodes.x);
    filt1 = filt1*filt1;
    filt1 = __frcp_rn(filt1 * filt1);
    if (kmodes.x == 0){
        filt1 = 1.0;
    }

    float filt2 = sinf(0.5f * kmodes.y) / (0.5 * kmodes.y);
    filt2 = filt2*filt2;
    filt2 = __frcp_rn(filt2 * filt2);
    if (kmodes.y == 0){
        filt2 = 1.0;
    }

    float filt3 = sinf(0.5f * kmodes.z) / (0.5 * kmodes.z);
    filt3 = filt3*filt3;
    filt3 = __frcp_rn(filt3 * filt3);
    if (kmodes.z == 0){
        filt3 = 1.0;
    }
    double filter = filt1 * filt2 * filt3;

    deviceFFT_t my_grid = __ldg(&grid[idx]);
    my_grid.x *= filter;
    grid[idx] = my_grid;
    
}

__global__ void foldParticles(float4* __restrict d_pos, double ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float4 my_particle = __ldg(&d_pos[idx]);
    my_particle.x /= ng;
    if (my_particle.x >= 0.5){
        my_particle.x -= 0.5;
    }
    my_particle.x *= 2 * ng;

    my_particle.y /= ng;
    if (my_particle.y >= 0.5){
        my_particle.y -= 0.5;
    }
    my_particle.y *= 2 * ng;

    my_particle.z /= ng;
    if (my_particle.z >= 0.5){
        my_particle.z -= 0.5;
    }
    my_particle.z *= 2 * ng;

    d_pos[idx] = my_particle;
}

__global__ void ScaleParticles(float4* __restrict d_pos, double oldNg, double newNg){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float4 my_particle = __ldg(&d_pos[idx]);
    float mul = newNg/oldNg;
    my_particle.x *= mul;
    my_particle.y *= mul;
    my_particle.z *= mul;
    d_pos[idx] = my_particle;
}

__global__ void cpy(float4* __restrict dest, const float4* __restrict source){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //printf("%g %g %g\n",source[idx],source[idx],source[idx]);
    dest[idx] = __ldg(&source[idx]);
}

__global__ void BinPower(const hostFFT_t* __restrict d_powerSpectrum, double* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 0){
        return;
    }
    if (idx == ((ng/2)*(ng)*(ng) + (ng/2)*ng + (ng/2))){
        return;
    }

    double d = ((2*M_PI)/(((double)(rl))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    hostFFT_t this_val = __ldg(&d_powerSpectrum[idx]);

    float kbin = sqrtf(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z) - minK;
    int indx = (int)(kbin/binDelta);

    atomicAdd(&d_binVals[indx],this_val);
    atomicAdd(&d_binCounts[indx],1);


}

__global__ void BinPower(const deviceFFT_t* __restrict d_grid, double* __restrict d_binVals, int* __restrict d_binCounts, double minK, double binDelta, double rl, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx == 0){
        return;
    }
    if (idx == ((ng/2)*(ng)*(ng) + (ng/2)*ng + (ng/2))){
        return;
    }

    double d = ((2*M_PI)/(((double)(rl))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    deviceFFT_t this_val = __ldg(&d_grid[idx]);

    float kbin = sqrtf(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z) - minK;
    int indx = (int)(kbin/binDelta);

    atomicAdd(&d_binVals[indx],this_val.x);
    atomicAdd(&d_binCounts[indx],1);


}

/*void HACCGPM::serial::GetFinerPowerSpectrum(float4* d_temp_pos, int ng, double rl, int nbins, int fftNg, const char* fname, int blockSize){
    int baseNumBlocks = (ng*ng*ng)/blockSize;
    int finalNumBlocks = (fftNg * fftNg * fftNg)/blockSize;

    float4* d_pos; cudaCall(cudaMalloc,&d_pos,sizeof(float4)*ng*ng*ng);
    deviceFFT_t* d_grid1; cudaCall(cudaMalloc,&d_grid1,sizeof(deviceFFT_t)*fftNg*fftNg*fftNg);
    deviceFFT_t* d_grid2; cudaCall(cudaMalloc,&d_grid2,sizeof(deviceFFT_t)*fftNg*fftNg*fftNg);
    hostFFT_t* d_powerSpectrum; cudaCall(cudaMalloc,&d_powerSpectrum,sizeof(hostFFT_t)*fftNg*fftNg*fftNg);

    hostFFT_t* tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*fftNg*fftNg*fftNg);

    double* kBins = (double*)malloc(sizeof(double)*nbins);
    int* binCounts = (int*)malloc(sizeof(int)*(nbins));
    double* binVals = (double*)malloc(sizeof(double)*nbins);

    //cudaCall(cudaMemset,d_grid2,0,sizeof(deviceFFT_t)*fftNg*fftNg*fftNg);
    InvokeGPUKernel(cpy,baseNumBlocks,blockSize,d_pos,d_temp_pos);

    //cudaCall(cudaMemcpy, d_pos, d_temp_pos, sizeof(float4)*ng*ng*ng, cudaMemcpyDeviceToDevice);

    InvokeGPUKernel(ScaleParticles,baseNumBlocks,blockSize,d_pos,(double)ng,(double)fftNg);

    //InvokeGPUKernel(HACCGPM::serial::CICKernel,baseNumBlocks,blockSize,d_grid2,d_pos,fftNg,1.0f);
    HACCGPM::serial::CIC(d_grid2,d_pos,fftNg,blockSize);

    HACCGPM::serial::forward_fft(d_grid2,d_grid1,fftNg);

    InvokeGPUKernel(scalePower,finalNumBlocks,blockSize,d_grid1,d_powerSpectrum,(double)ng,rl);

    InvokeGPUKernel(PkCICFilter,finalNumBlocks,blockSize,d_powerSpectrum,fftNg,0);

    cudaCall(cudaMemcpy, tmp, d_powerSpectrum, sizeof(hostFFT_t)*fftNg*fftNg*fftNg, cudaMemcpyDeviceToHost);

    double d = (2*M_PI)/(rl);
    
    double minK = d;

    double maxK = sqrt(3*((fftNg/2)*(fftNg/2)*d*d));

    double binDelta = (maxK - minK)/((double)nbins - 2);

    minK -= binDelta * 0.5f;

    //printf("maxK %g, binDelta %g\n",maxK,binDelta);

    for (int i = 0; i < nbins; i++){
        kBins[i] = minK + binDelta * (i+0.5f);
        binCounts[i] = 0;
        binVals[i] = 0;
    }


    for (int i = 0; i < fftNg; i++){
            double l = i;
            if (i > ((fftNg/2)-1)){
                l = -(fftNg - i);
            }
            l *= d;
            for (int j = 0; j < fftNg; j++){
                double m = j;
                if (j > ((fftNg/2)-1)){
                    m = -(fftNg - j);
                }
                m *= d;
                for (int k = 0; k < fftNg; k++){
                    double n = k;
                    if (k > ((fftNg/2)-1)){
                        n = -(fftNg - k);
                    }
                    n *= d;
                    double this_val = tmp[i*fftNg*fftNg + j*fftNg + k];
                    double kbin = sqrt(l*l + m*m + n*n) - minK;
                    if ((kbin >= 0) && ((l > 0) && (m > 0) && (n > 0))){
                        int idx = (int)(kbin/binDelta);
                        if (idx >= nbins){
                            printf("I SHOULD NEVER HAPPEN SOMETHING HAS GONE TERRIBLY WRONG IN GET POWER SPECTRUM\n");
                        }
                        binVals[idx] += this_val;
                        binCounts[idx]++;
                    }
                }
            }
        }

    FILE *fp;
    fp = fopen(fname, "w+");
    for (int i = 0; i < nbins; i++){
        //printf("bin %d, count: %d, %g\n",i, binCounts[i], kBins[i]);
        double out = binVals[i]/((double)binCounts[i]);
        if ((out == out) && (binCounts[i] != 0) && (out != 0)){
            fprintf(fp,"%g,%g,%d\n",kBins[i],out,binCounts[i]);
        }
    }
    fclose(fp);
    
    cudaFree(d_pos);
    cudaFree(d_grid1);
    cudaFree(d_grid2);
    cudaFree(d_powerSpectrum);
    free(tmp);
    free(kBins);
    free(binCounts);
    free(binVals);

}*/

void HACCGPM::serial::GetPowerSpectrum(float4* d_pos, deviceFFT_t* d_grid, int ng, double rl, int nbins, const char* fname, int nfolds, int blockSize, int calls){

    CPUTimer_t start = CPUTimer();

    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);
    #ifdef VerbosePower
    printf("%sGetPowerSpectrum was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Allocating d_temp_pos, kBins, binCounts, binVals, d_binCounts, d_binVals\n",indent);
    #endif

    float4* d_temp_pos; cudaCall(cudaMalloc,&d_temp_pos,sizeof(float4)*ng*ng*ng);
    //hostFFT_t* d_powerSpectrum; cudaCall(cudaMalloc,&d_powerSpectrum,sizeof(hostFFT_t)*ng*ng*ng);

    double* kBins = (double*)malloc(sizeof(double)*nbins);
    int* binCounts = (int*)malloc(sizeof(int)*(nbins));
    double* binVals = (double*)malloc(sizeof(double)*nbins);

    int* d_binCounts; cudaCall(cudaMalloc,&d_binCounts,sizeof(int)*nbins);
    double* d_binVals; cudaCall(cudaMalloc,&d_binVals,sizeof(double)*nbins);

    //hostFFT_t* tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*ng*ng*ng);

    POWER_KERNEL_TIME += InvokeGPUKernel(cpy,numBlocks,blockSize,d_temp_pos,d_pos);

    double new_rl = (rl/(pow(2,nfolds)));

    double d = (2*M_PI)/(new_rl);
    double d3 = sqrt(3*d*d);
    
    double minK = d;

    double maxK = sqrt(3*((ng/2)*(ng/2)*d*d));

    double binDelta = (maxK - minK)/((double)nbins);

    for (int i = 0; i < nbins; i++){
        kBins[i] = minK + binDelta * (((float)i) + 0.5f);
    }

    new_rl = rl;

    for (int current_fold = 0; current_fold < nfolds + 1; current_fold++){

        cudaCall(cudaMemset,d_binCounts,0,sizeof(int)*nbins);
        cudaCall(cudaMemset,d_binVals,0,sizeof(double)*nbins);

        new_rl = (rl/(pow(2,current_fold)));

        d = (2*M_PI)/(new_rl);

        if (current_fold > 0){
            POWER_KERNEL_TIME += InvokeGPUKernel(foldParticles,numBlocks,blockSize,d_temp_pos,(double)ng);
        }

        CPUTimer_t start_extras = CPUTimer();

        HACCGPM::serial::CIC(d_grid,d_temp_pos,ng,blockSize,calls+1);

        HACCGPM::serial::forward_fft(d_grid,ng,calls + 1);

        CPUTimer_t end_extras = CPUTimer();

        POWER_KERNEL_TIME += end_extras - start_extras;

        POWER_KERNEL_TIME += InvokeGPUKernel(scalePower,numBlocks,blockSize,d_grid,(double)ng,rl);

        POWER_KERNEL_TIME += InvokeGPUKernel(PkCICFilter,numBlocks,blockSize,d_grid,ng,current_fold);

        POWER_KERNEL_TIME += InvokeGPUKernel(BinPower,numBlocks,blockSize,d_grid,d_binVals,d_binCounts,minK,binDelta,new_rl,ng);

        cudaCall(cudaMemcpy, binVals, d_binVals, sizeof(double)*nbins, cudaMemcpyDeviceToHost);
        cudaCall(cudaMemcpy, binCounts, d_binCounts, sizeof(int)*nbins, cudaMemcpyDeviceToHost);
 
        char fname2[400];
        sprintf(fname2, "%s.fold.%d", fname, current_fold);
        FILE *fp;
        fp = fopen(fname2, "w+");
        for (int i = 0; i < nbins; i++){
            double out = binVals[i]/((double)binCounts[i]);
            if ((out == out) && (binCounts[i] != 0) && (out != 0)){
                fprintf(fp,"%g,%g,%d\n",kBins[i],out,binCounts[i]);
            }
            binVals[i] = 0;
            binCounts[i] = 0;
        }
        fclose(fp);

    }
    free(kBins);
    free(binCounts);
    free(binVals);
    cudaCall(cudaFree,d_temp_pos);
    cudaCall(cudaFree,d_binCounts);
    cudaCall(cudaFree,d_binVals);

    CPUTimer_t end = CPUTimer();

    CPUTimer_t t = end-start;
    POWER_CALLS++;
    POWER_TIME += t;

    printf("%s   GetPowerSpectrum took %llu us\n",indent,t);

}

void HACCGPM::serial::printPowerTimes(){
    printf("   GetPowerSpectrum  -> calls: %10d | total: %10llu us | cpu: %10llu us | gpu: %10llu us | mean: %10.2f us\n",POWER_CALLS,POWER_TIME,POWER_TIME - POWER_KERNEL_TIME,POWER_KERNEL_TIME,((float)POWER_TIME)/((float)(POWER_CALLS)));
}