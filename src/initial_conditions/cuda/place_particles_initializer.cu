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

void launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    InvokeGPUKernel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng);
}

void launch_place_particles(float4* d_pos, float4* d_vel, floatFFT_t* d_x, floatFFT_t* d_y, floatFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    InvokeGPUKernel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng);
}

void launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int nlocal, int3 local_grid_size, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    InvokeGPUKernelParallel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng,local_grid_size.x,local_grid_size.y,local_grid_size.z, nlocal, world_rank);
}