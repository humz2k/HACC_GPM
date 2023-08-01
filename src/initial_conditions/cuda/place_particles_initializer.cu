#include "../ic_kernels.hpp"

template<class T>
__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, T* __restrict outSx, T* __restrict outSy, T* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float4 my_particle = make_float4(idx3d.x,idx3d.y,idx3d.z,idx);

    T thisSx = __ldg(&outSx[idx]);
    T thisSy = __ldg(&outSy[idx]);
    T thisSz = __ldg(&outSz[idx]);

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

template<class T>
__global__ void placeParticles(float3* __restrict d_pos, float3* __restrict d_vel, T* __restrict outSx, T* __restrict outSy, T* __restrict outSz, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float3 my_particle = make_float3(idx3d.x,idx3d.y,idx3d.z);

    T thisSx = __ldg(&outSx[idx]);
    T thisSy = __ldg(&outSy[idx]);
    T thisSz = __ldg(&outSz[idx]);

    float3 s = make_float3(thisSx.x,thisSy.x,thisSz.x);

    float velA = a - (deltaT * 0.5f);
    float velMul = (velA * velA * dotDelta * fscal);
    float3 my_vel = make_float3(velMul*s.x,velMul*s.y,velMul*s.z);
    my_particle.x += delta * s.x + ng;
    my_particle.x = fmod(my_particle.x,(float)ng);
    my_particle.y += delta * s.y + ng;
    my_particle.y = fmod(my_particle.y,(float)ng);
    my_particle.z += delta * s.z + ng;
    my_particle.z = fmod(my_particle.z,(float)ng);

    d_pos[idx] = my_particle;
    d_vel[idx] = my_vel;
}

template<class T>
__global__ void placeParticles(float4* __restrict d_pos, float4* __restrict d_vel, T* __restrict outS, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float4 my_particle = make_float4(idx3d.x,idx3d.y,idx3d.z,idx);

    T s = __ldg(&outS[idx]);

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

template<class T>
__global__ void placeParticles(float3* __restrict d_pos, float3* __restrict d_vel, T* __restrict outS, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);

    float3 my_particle = make_float3(idx3d.x,idx3d.y,idx3d.z);

    T s = __ldg(&outS[idx]);

    float velA = a - (deltaT * 0.5f);
    float velMul = (velA * velA * dotDelta * fscal);
    float3 my_vel = make_float3(velMul*s.x,velMul*s.y,velMul*s.z);
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

template<class T>
CPUTimer_t launch_place_particles(T* d_pos, T* d_vel, float4* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_grad,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng);
}

template CPUTimer_t launch_place_particles<float4>(float4*,float4*,float4*,double,double,double,double,double,double,int,int,int,int);
template CPUTimer_t launch_place_particles<float3>(float3*,float3*,float4*,double,double,double,double,double,double,int,int,int,int);

template<class T1, class T2>
CPUTimer_t launch_place_particles(T1* d_pos, T1* d_vel, T2* d_x, T2* d_y, T2* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng);
}

template CPUTimer_t launch_place_particles<float4,deviceFFT_t>(float4*,float4*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,double,double,double,double,double,double,int,int,int,int);
template CPUTimer_t launch_place_particles<float3,deviceFFT_t>(float3*,float3*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,double,double,double,double,double,double,int,int,int,int);
template CPUTimer_t launch_place_particles<float4,floatFFT_t>(float4*,float4*,floatFFT_t*,floatFFT_t*,floatFFT_t*,double,double,double,double,double,double,int,int,int,int);
template CPUTimer_t launch_place_particles<float3,floatFFT_t>(float3*,float3*,floatFFT_t*,floatFFT_t*,floatFFT_t*,double,double,double,double,double,double,int,int,int,int);


CPUTimer_t launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int nlocal, int3 local_grid_size, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng,local_grid_size.x,local_grid_size.y,local_grid_size.z, nlocal, world_rank);
}