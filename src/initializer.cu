#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../cambTools/ccamb.h"

#define VerboseInitializer

//#define NOPYTHON

__global__ void initRNG(curandState *state, int seed){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);

}

__global__ void initRNG(curandState *state, int seed, int nlocal, int world_rank){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx >= nlocal)return;
  curand_init(seed, idx + nlocal * world_rank, 0, &state[idx]);

}

__global__ void GenerateRealRandom(curandState* state, deviceFFT_t* __restrict grid){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    hostFFT_t amp = curand_normal_double(state + idx);
    deviceFFT_t out;
    out.x = amp;
    out.y = 0;
    grid[idx] = out;
}

__global__ void GenerateRealRandom(curandState* state, deviceFFT_t* __restrict grid, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t amp = curand_normal_double(state + idx);
    deviceFFT_t out;
    out.x = amp;
    out.y = 0;
    grid[idx] = out;
}

__global__ void ScaleAmplitudes(deviceFFT_t* __restrict grid, const hostFFT_t* __restrict scale){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    deviceFFT_t current = __ldg(&grid[idx]);
    //printf("d_grid[%d]: %g %g\n",idx,current.x,current.y);
    current.x *= scale_by;
    current.y *= scale_by;
    //printf("%g %g\n",current.x,current.y);
    grid[idx] = current;
}

__global__ void ScaleAmplitudes(deviceFFT_t* __restrict grid, const hostFFT_t* __restrict scale, int nlocal){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= nlocal)return;
    hostFFT_t base_scale_by = __ldg(&scale[idx]);
    hostFFT_t scale_by = sqrt(base_scale_by);
    deviceFFT_t current = __ldg(&grid[idx]);
    //printf("d_grid[%d]: %g %g\n",idx,current.x,current.y);
    current.x *= scale_by;
    current.y *= scale_by;
    //printf("%g %g\n",current.x,current.y);
    grid[idx] = current;
}

__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    deviceFFT_t old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}

__global__ void ScaleFFT(deviceFFT_t* __restrict data, double scale, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;

    deviceFFT_t old = __ldg(&data[idx]);

    old.x *= scale;
    old.y *= scale;

    data[idx] = old;

}

__global__ void transformDensityField(const deviceFFT_t* __restrict oldGrid, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double rl, double a, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    double k2 = kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z;

    double k2mul = (1/k2);
    if (k2 == 0){
        k2mul = 0;
    }

    double mul = (1/delta) * k2mul;

    deviceFFT_t current = __ldg(&oldGrid[idx]);
    current.x *= mul;
    current.y *= mul;

    deviceFFT_t sx,sy,sz;

    sx.x = current.y * kmodes.x;
    sx.y = -current.x * kmodes.x;

    sy.x = current.y * kmodes.y;
    sy.y = -current.x * kmodes.y;

    sz.x = current.y * kmodes.z;
    sz.y = -current.x * kmodes.z;

    outSx[idx] = sx;
    outSy[idx] = sy;
    outSz[idx] = sz;

}

__global__ void transformDensityField(const deviceFFT_t* __restrict oldGrid, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double rl, double a, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords,dims);
    float3 kmodes = HACCGPM::parallel::get_kmodes(idx3d,ng,d);

    double k2 = kmodes.x * kmodes.x + kmodes.y * kmodes.y + kmodes.z * kmodes.z;

    double k2mul = (1/k2);
    if (k2 == 0){
        k2mul = 0;
    }

    double mul = (1/delta) * k2mul;

    deviceFFT_t current = __ldg(&oldGrid[idx]);
    current.x *= mul;
    current.y *= mul;

    deviceFFT_t sx,sy,sz;

    sx.x = current.y * kmodes.x;
    sx.y = -current.x * kmodes.x;

    sy.x = current.y * kmodes.y;
    sy.y = -current.x * kmodes.y;

    sz.x = current.y * kmodes.z;
    sz.y = -current.x * kmodes.z;

    outSx[idx] = sx;
    outSy[idx] = sy;
    outSz[idx] = sz;

}

__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float4 my_particle = make_float4(idx3d.x,idx3d.y,idx3d.z,idx);

    deviceFFT_t thisSx = __ldg(&outSx[idx]);
    deviceFFT_t thisSy = __ldg(&outSy[idx]);
    deviceFFT_t thisSz = __ldg(&outSz[idx]);

    float3 s = make_float3(thisSx.x,thisSy.x,thisSz.x);

    float velA = a - (deltaT * 0.5f);
    float velMul = (velA * velA * dotDelta * fscal);
    float4 my_vel = make_float4(velMul*s.x,velMul*s.y,velMul*s.z,idx);
    my_particle.x += delta * s.x + ng;
    my_particle.x = fmod(my_particle.x,(float)ng);
    my_particle.y += delta * s.y + ng;
    my_particle.y = fmod(my_particle.y,(float)ng);
    my_particle.z += delta * s.z + ng;
    my_particle.z = fmod(my_particle.z,(float)ng);

    d_pos[idx] = my_particle;
    d_vel[idx] = my_vel;
}

__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, deviceFFT_t* __restrict outSx, deviceFFT_t* __restrict outSy, deviceFFT_t* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng, int nx, int ny, int nz, int nlocal, int world_rank){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    int3 idx3d = HACCGPM::parallel::get_local_index(idx,nx,ny,nz);

    float4 my_particle = make_float4(idx3d.x,idx3d.y,idx3d.z,idx + nlocal*world_rank);

    deviceFFT_t thisSx = __ldg(&outSx[idx]);
    deviceFFT_t thisSy = __ldg(&outSy[idx]);
    deviceFFT_t thisSz = __ldg(&outSz[idx]);

    float3 s = make_float3(thisSx.x,thisSy.x,thisSz.x);

    float velA = a - (deltaT * 0.5f);
    float velMul = (velA * velA * dotDelta * fscal);
    float4 my_vel = make_float4(velMul*s.x,velMul*s.y,velMul*s.z,idx + nlocal*world_rank);

    my_particle.x += delta * s.x;
    my_particle.y += delta * s.y;
    my_particle.z += delta * s.z;

    d_pos[idx] = my_particle;
    d_vel[idx] = my_vel;
}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0){
        out[idx] = 0;
        return;
    }

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::serial::get_kmodes(idx3d,ng,d);

    double my_k = sqrt(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z);

    int left_bin = (int)(my_k / k_delta);
    int right_bin = left_bin + 1;

    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(my_k);
    //double frac = 1.0f/abs(logx1 - logx2);
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        //printf("%d -> %d: %g -> %g\n",left_bin,right_bin,in[left_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)),in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)));
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }
    //printf("%g: %g > %g > %g\n",my_k,exp(logy1)*(((double)(ng*ng*ng))/(rl*rl*rl)),y,exp(logy2)*(((double)(ng*ng*ng))/(rl*rl*rl)));
    //printf("x1,y1=%g,%g, x2,y2=%g,%g\n",logx1,logy1,logx2,logy2);
    out[idx] = (y);

}

__global__ void interpolatePowerSpectrum(hostFFT_t* out, double* in, int nbins, double k_delta, double k_min, double rl, int ng, int nlocal, int world_rank, int3 local_grid_size, int3 local_coords, int3 dims){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    if (idx == 0){
        out[idx] = 0;
        return;
    }

    double d = (2*M_PI)/rl;

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords,dims);
    float3 kmodes = HACCGPM::parallel::get_kmodes(idx3d,ng,d);

    double my_k = sqrt(kmodes.x*kmodes.x + kmodes.y*kmodes.y + kmodes.z*kmodes.z);

    int left_bin = (int)(my_k / k_delta);
    int right_bin = left_bin + 1;

    double logy1 = log(in[left_bin]);
    double logx1 = log(k_delta * (double)left_bin);
    double logy2 = log(in[right_bin]);
    double logx2 = log(k_delta * (double)right_bin);
    double logx = log(my_k);
    //double frac = 1.0f/abs(logx1 - logx2);
    double logy = logy1 + ((logy2 - logy1)/(logx2 - logx1)) * (logx - logx1);
    double y = exp(logy) * (((double)(ng*ng*ng))/(rl*rl*rl));
    if (left_bin == 0){
        //printf("%d -> %d: %g -> %g\n",left_bin,right_bin,in[left_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)),in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl)));
        y = in[right_bin] * (((double)(ng*ng*ng))/(rl*rl*rl));
    }
    //printf("%g: %g > %g > %g\n",my_k,exp(logy1)*(((double)(ng*ng*ng))/(rl*rl*rl)),y,exp(logy2)*(((double)(ng*ng*ng))/(rl*rl*rl)));
    //printf("x1,y1=%g,%g, x2,y2=%g,%g\n",logx1,logy1,logx2,logy2);
    out[idx] = (y);

}

void GenerateFourierAmplitudes(const char* params_file, HACCGPM::CosmoClass& cosmo, deviceFFT_t* d_grid1, int ng, double rl, double z, int seed, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateFourierAmplitudes was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,blockSize,indent,numBlocks,indent,z);
    printf("%s   Allocating rngState, h_tmp, d_pkScale...\n",indent);
    #endif

    curandState* rngState; cudaCall(cudaMalloc,&rngState,sizeof(curandState)*ng*ng*ng);
    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*ng*ng*ng);
    hostFFT_t* d_pkScale; cudaCall(cudaMalloc,&d_pkScale,sizeof(hostFFT_t)*ng*ng*ng);

    #ifdef VerboseInitializer
    printf("%s      Allocated rngState, h_tmp, d_pkScale.\n",indent);
    printf("%s   Calling initRNG...\n",indent);
    #endif

    InvokeGPUKernel(initRNG,numBlocks,blockSize,rngState,seed);

    #ifdef VerboseInitializer
    printf("%s      Called initRNG.\n",indent);
    printf("%s   Calling GenerateRealRandom...\n",indent);
    #endif

    InvokeGPUKernel(GenerateRealRandom,numBlocks,blockSize,rngState,d_grid1);

    #ifdef VerboseInitializer
    printf("%s      Called GenerateRealRandom.\n",indent);
    printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::serial::forward_fft(d_grid1,ng,calls+1);

    #ifdef VerboseInitializer
    printf("%s      Done Forward FFT...\n",indent);
    
    #endif

    //init_python(calls + 1);
    #ifdef NOPYTHON
    #ifdef VerboseInitializer
    printf("%s   Getting Pk from ipk...\n",indent);
    #endif
    double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((ng/2)*2*M_PI)/rl;
    maxK = sqrt(maxK*maxK*maxK);
    printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }

    InvokeGPUKernel(interpolatePowerSpectrum,numBlocks,blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,rl,ng);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);
    #else
    #ifdef VerboseInitializer
    printf("%s   Getting Pk from Camb...\n",indent);
    #endif
    get_pk(params_file,h_tmp,z,ng,rl,calls+1);

    #ifdef VerboseInitializer
    printf("%s      Got Pk from Camb.\n",indent);
    printf("%s   Copying Pk from host to device...\n",indent);
    #endif
    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*ng*ng*ng, cudaMemcpyHostToDevice);
    
    #ifdef VerboseInitializer
    printf("%s      Copied Pk from host to device.\n",indent);
    #endif

    #endif
    
    
    #ifdef VerboseInitializer
    printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    InvokeGPUKernel(ScaleAmplitudes,numBlocks,blockSize,d_grid1,d_pkScale);

    #ifdef VerboseInitializer
    printf("%s      Scaled Amplitudes.\n",indent);
    printf("%s   Freeing rngState, h_tmp, d_pkScale...\n",indent);
    #endif

    free(h_tmp);
    cudaFree(d_pkScale);
    cudaFree(rngState);

    #ifdef VerboseInitializer
    printf("%s      Freed rngState, h_tmp, d_pkScale.\n",indent);
    #endif
}

void HACCGPM::serial::GenerateDisplacementIC(const char* params_file, HACCGPM::serial::MemoryManager* mem, HACCGPM::CosmoClass& cosmo, int ng, double rl, double z, double deltaT, double fscal, int seed, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,blockSize,indent,numBlocks,indent,params_file);
    printf("%s   Calling GenerateFourierAmplitudes...\n",indent);
    #endif

    GenerateFourierAmplitudes(params_file, cosmo, mem->d_grid, ng, rl, z, seed, blockSize, calls+1);

    #ifdef VerboseInitializer
    printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    printf("%s   Allocating d_sx, d_sy, d_sz...\n",indent);
    #endif

    deviceFFT_t* d_sx; cudaCall(cudaMalloc,&d_sx,sizeof(deviceFFT_t)*ng*ng*ng);
    deviceFFT_t* d_sy; cudaCall(cudaMalloc,&d_sy,sizeof(deviceFFT_t)*ng*ng*ng);
    deviceFFT_t* d_sz; cudaCall(cudaMalloc,&d_sz,sizeof(deviceFFT_t)*ng*ng*ng);

    #ifdef VerboseInitializer
    printf("%s      Allocated d_sx, d_sy, d_sz.\n",indent);
    printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(z + 1)) - (deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;
    //get_delta_and_dotDelta(params_file, z,velZ,&delta,&dotDelta,calls+1);
    cosmo.get_delta_and_dotDelta(z,velZ,&delta,&dotDelta);
    
    printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    //finalize_python(calls+1);

    #ifdef VerboseInitializer
    printf("%s      Called get_delta_and_dotDelta.\n",indent);
    printf("%s   Calling transformDensityField...\n",indent);
    #endif

    InvokeGPUKernel(transformDensityField,numBlocks,blockSize,mem->d_grid,d_sx,d_sy,d_sz,delta,rl,1/(1+z),ng);

    #ifdef VerboseInitializer
    printf("%s      Called transformVelocityField.\n",indent);
    printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(d_sx,ng,calls+1);
    HACCGPM::serial::backward_fft(d_sy,ng,calls+1);
    HACCGPM::serial::backward_fft(d_sz,ng,calls+1);

    double scale_by = 1.0f/((double)(ng*ng*ng));

    InvokeGPUKernel(ScaleFFT,numBlocks,blockSize,d_sx,scale_by);
    InvokeGPUKernel(ScaleFFT,numBlocks,blockSize,d_sy,scale_by);
    InvokeGPUKernel(ScaleFFT,numBlocks,blockSize,d_sz,scale_by);

    #ifdef VerboseInitializer
    printf("%s      Done Backward FFTs.\n",indent);
    #endif

    InvokeGPUKernel(placeParticles,numBlocks,blockSize,mem->d_pos,mem->d_vel,d_sx,d_sy,d_sz,delta,dotDelta,rl,1/(1+z),deltaT,fscal,ng);

    #ifdef VerboseInitializer
    printf("%s   Freeing d_sx, d_sy, d_sz...\n",indent);
    #endif

    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_sz);

    #ifdef VerboseInitializer
    printf("%s      Freed d_sx, d_sy, d_sz.\n",indent);
    #endif
}


void GenerateFourierAmplitudesParallel(const char* params_file, HACCGPM::CosmoClass& cosmo, deviceFFT_t* d_grid, int ng, double rl, double z, int seed, int blockSize, int world_rank, int world_size, int nlocal, int* local_grid_size, int* local_coords, int* dims, int calls){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;

    getIndent(calls);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%sGenerateFourierAmplitudesParallel was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   z %g\n",indent,indent,blockSize,indent,numBlocks,indent,z);
    if(world_rank == 0)printf("%s   Allocating rngState, h_tmp, d_pkScale...\n",indent);
    #endif

    curandState* rngState; cudaCall(cudaMalloc,&rngState,sizeof(curandState)*nlocal);
    hostFFT_t* h_tmp = (hostFFT_t*)malloc(sizeof(hostFFT_t)*nlocal);
    hostFFT_t* d_pkScale; cudaCall(cudaMalloc,&d_pkScale,sizeof(hostFFT_t)*nlocal);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Allocated rngState, h_tmp, d_pkScale.\n",indent);
    if(world_rank == 0)printf("%s   Calling initRNG...\n",indent);
    #endif

    InvokeGPUKernelParallel(initRNG,numBlocks,blockSize,rngState,seed,nlocal,world_rank);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Called initRNG.\n",indent);
    if(world_rank == 0)printf("%s   Calling GenerateRealRandom...\n",indent);
    #endif

    InvokeGPUKernelParallel(GenerateRealRandom,numBlocks,blockSize,rngState,d_grid,nlocal);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Called GenerateRealRandom.\n",indent);
    if(world_rank == 0)printf("%s   Doing Forward FFT...\n",indent);
    #endif

    HACCGPM::parallel::forward_fft(d_grid,ng,calls+1);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Done Forward FFT...\n",indent);
    #endif
    #ifdef NOPYTHON
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from ipk...\n",indent);
    #endif
    double* h_ipk;
    int ipk_bins;
    double ipk_delta;
    double ipk_max;
    double ipk_min;
    cosmo.read_ipk(&h_ipk,&ipk_bins,&ipk_delta,&ipk_max,&ipk_min,calls+1);
    double* d_ipk; cudaCall(cudaMalloc,&d_ipk,sizeof(double)*ipk_bins);
    cudaCall(cudaMemcpy, d_ipk, h_ipk, sizeof(double)*ipk_bins, cudaMemcpyHostToDevice);

    double maxK = ((ng/2)*2*M_PI)/rl;
    maxK = sqrt(maxK*maxK*maxK);
    if(world_rank == 0)printf("%s      maxK = %g\n",indent,maxK);
    if (maxK > ipk_max){
        if(world_rank == 0)printf("%s      input ipk only goes to %g\n",indent,ipk_max);
        exit(1);
    }
    int3 local_grid_size_vec = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    int3 dims_vec = make_int3(dims[0],dims[1],dims[2]);
    int3 local_coords_vec = make_int3(local_coords[0],local_coords[1],local_coords[2]);
    InvokeGPUKernelParallel(interpolatePowerSpectrum,numBlocks,blockSize,d_pkScale,d_ipk,ipk_bins,ipk_delta,ipk_min,rl,ng,nlocal,world_rank,local_grid_size_vec,local_coords_vec,dims_vec);

    free(h_ipk);
    cudaCall(cudaFree,d_ipk);
    #else
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Getting Pk from Camb...\n",indent);
    #endif
    get_pk_parallel(params_file,h_tmp,z,ng,rl,nlocal,world_rank,calls+1);
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Got Pk from Camb.\n",indent);
    #endif
    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Copying Pk from host to device...\n",indent);
    #endif

    cudaCall(cudaMemcpy, d_pkScale, h_tmp, sizeof(hostFFT_t)*nlocal, cudaMemcpyHostToDevice);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Copied Pk from host to device.\n",indent);
    if(world_rank == 0)printf("%s   Scaling Amplitudes...\n",indent);
    #endif
    #endif

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s   Scaling Amplitudes...\n",indent);
    #endif

    InvokeGPUKernelParallel(ScaleAmplitudes,numBlocks,blockSize,d_grid,d_pkScale,nlocal);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Scaled Amplitudes.\n",indent);
    if(world_rank == 0)printf("%s   Freeing rngState, h_tmp, d_pkScale...\n",indent);
    #endif

    free(h_tmp);
    cudaFree(d_pkScale);
    cudaFree(rngState);

    #ifdef VerboseInitializer
    if(world_rank == 0)printf("%s      Freed rngState, h_tmp, d_pkScale.\n",indent);
    #endif
}

void HACCGPM::parallel::GenerateDisplacementIC(const char* params_file, HACCGPM::parallel::MemoryManager* mem, HACCGPM::CosmoClass& cosmo, int ng, double rl, double z, double deltaT, double fscal, int seed, int blockSize, int world_rank, int world_size, int nlocal, int* local_grid_size, int* local_coords, int* dims, int calls){

    int numBlocks = (nlocal)/blockSize;
    getIndent(calls);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%sGenerateDisplacementIC was called with\n%s   blockSize %d\n%s   numBlocks %d\n%s   params %s\n",indent,indent,blockSize,indent,numBlocks,indent,params_file);
    if (world_rank == 0)printf("%s   Calling GenerateFourierAmplitudesParallel...\n",indent);
    #endif

    GenerateFourierAmplitudesParallel(params_file, cosmo, mem->d_grid, ng, rl, z, seed, blockSize, world_rank, world_size, nlocal, local_grid_size, local_coords, dims, calls+1);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Called GenerateFourierAmplitudes.\n",indent);
    if (world_rank == 0)printf("%s   Allocating d_sx, d_sy, d_sz...\n",indent);
    #endif

    deviceFFT_t* d_sx; cudaCall(cudaMalloc,&d_sx,sizeof(deviceFFT_t)*nlocal);
    deviceFFT_t* d_sy; cudaCall(cudaMalloc,&d_sy,sizeof(deviceFFT_t)*nlocal);
    deviceFFT_t* d_sz; cudaCall(cudaMalloc,&d_sz,sizeof(deviceFFT_t)*nlocal);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Allocated d_sx, d_sy, d_sz.\n",indent);
    if (world_rank == 0)printf("%s   Calling get_delta_and_dotDelta...\n",indent);
    #endif

    double delta;
    double dotDelta;
    double this_a = (1/(z + 1)) - (deltaT/2.0f);
    double velZ = (1.0f/this_a) - 1.0f;
    cosmo.get_delta_and_dotDelta(z,velZ,&delta,&dotDelta);

    if (world_rank == 0)printf("%s      Delta %g, dotDelta %g\n",indent,delta,dotDelta);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Called get_delta_and_dotDelta.\n",indent);
    if (world_rank == 0)printf("%s   Calling transformDensityField...\n",indent);
    #endif
    int3 local_grid_size_vec = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    int3 dims_vec = make_int3(dims[0],dims[1],dims[2]);
    int3 local_coords_vec = make_int3(local_coords[0],local_coords[1],local_coords[2]);
    InvokeGPUKernelParallel(transformDensityField,numBlocks,blockSize,mem->d_grid,d_sx,d_sy,d_sz,delta,rl,1/(1+z),ng,nlocal,world_rank,local_grid_size_vec,local_coords_vec,dims_vec);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Called transformVelocityField.\n",indent);
    if (world_rank == 0)printf("%s   Doing Backward FFTs...\n",indent);
    #endif

    HACCGPM::parallel::backward_fft(d_sx,ng,calls+1);
    HACCGPM::parallel::backward_fft(d_sy,ng,calls+1);
    HACCGPM::parallel::backward_fft(d_sz,ng,calls+1);

    double scale_by = 1.0f/((double)(ng*ng*ng));

    InvokeGPUKernelParallel(ScaleFFT,numBlocks,blockSize,d_sx,scale_by,nlocal);
    InvokeGPUKernelParallel(ScaleFFT,numBlocks,blockSize,d_sy,scale_by,nlocal);
    InvokeGPUKernelParallel(ScaleFFT,numBlocks,blockSize,d_sz,scale_by,nlocal);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Done Backward FFTs.\n",indent);
    #endif

    InvokeGPUKernelParallel(placeParticles,numBlocks,blockSize,mem->d_pos,mem->d_vel,d_sx,d_sy,d_sz,delta,dotDelta,rl,1/(1+z),deltaT,fscal,ng,local_grid_size[0],local_grid_size[1],local_grid_size[2], nlocal, world_rank);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s   Freeing d_sx, d_sy, d_sz...\n",indent);
    #endif

    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_sz);

    #ifdef VerboseInitializer
    if (world_rank == 0)printf("%s      Freed d_sx, d_sy, d_sz.\n",indent);
    #endif

}