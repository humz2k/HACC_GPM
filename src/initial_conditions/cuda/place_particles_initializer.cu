#include "../ic_kernels.hpp"

__forceinline__ __device__ float3 as_float3(float3 a){
    return a;
}

__forceinline__ __device__ float3 as_float3(float4 a){
    return make_float3(a.x,a.y,a.z);
}

__forceinline__ __device__ float4 join(float3 a, float s){
    return make_float4(a.x,a.y,a.z,s);
}

__forceinline__ __device__ void assign(float4 a, float4& out){
    out = a;
}

__forceinline__ __device__ void assign(float4 a, float3& out){
    out = as_float3(a);
}

template<class T, class T1>
__forceinline__ __device__ void init_vel(T s, double a, double deltaT, double dotDelta, double fscal, int idx, T1& out){
    float velA = a - (deltaT * 0.5f);
    float velMul = (velA * velA * dotDelta * fscal);
    float4 my_vel = join(velMul * as_float3(s),idx);
    assign(my_vel,out);
}

template<class T, class T1>
__forceinline__ __device__ void init_pos(T s, int3 idx3d, double delta, float ng, float np, int idx, T1& out){
    float4 my_particle = join(fmod(((make_float3(idx3d.x,idx3d.y,idx3d.z) + (delta * as_float3(s))) * (ng/np) + ng),ng),idx);
    assign(my_particle,out);
}

template<class T1, class T2>
__forceinline__ __device__ void init_particle(T1* __restrict d_pos, T1* __restrict d_vel, T2 s, int3 idx3d, int idx, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng, int np){
    T1 my_particle; init_pos(s,idx3d,delta,ng,np,idx,my_particle);
    T1 my_vel; init_vel(s,a,deltaT,dotDelta,fscal,idx,my_vel);

    d_pos[idx] = my_particle;// * ((float)(ng/np));
    d_vel[idx] = my_vel * (((float)ng/(float)np));

    //printf("%f %f %f\n",my_particle.x,my_particle.y,my_particle.z);

}

template<class T1, class T2>
__global__ void placeParticles(T1* __restrict d_pos, T1* __restrict d_vel, T2* __restrict outS, double delta, double dotDelta, double rl, double a, double deltaT, double fscal, int ng, int np){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= np*np*np)return;

    int3 idx3d = HACCGPM::serial::get_index(idx,np);

    //printf("idx3d: %d %d %d\n",idx3d.x,idx3d.y,idx3d.z);

    //int s_idx = idx3d.x * ng * ng + idx3d.y * ng + idx3d.z;

    T2 s = __ldg(&outS[idx]);

    init_particle(d_pos,d_vel,s,idx3d,idx,delta,dotDelta,rl,a,deltaT,fscal,ng,np);
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
CPUTimer_t launch_place_particles(T* d_pos, T* d_vel, float4* d_grad, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int np, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_grad,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng,np);
}

template CPUTimer_t launch_place_particles<float4>(float4*,float4*,float4*,double,double,double,double,double,double,int,int,int,int,int);
template CPUTimer_t launch_place_particles<float3>(float3*,float3*,float4*,double,double,double,double,double,double,int,int,int,int,int);


CPUTimer_t launch_place_particles(float4* d_pos, float4* d_vel, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, double delta, double dotDelta, double rl, double z_ini, double deltaT, double fscal, int ng, int nlocal, int3 local_grid_size, int world_rank, int numBlocks, int blockSize, int calls){
    getIndent(calls);
    return InvokeGPUKernelParallel(placeParticles,numBlocks,blockSize,d_pos,d_vel,d_x,d_y,d_z,delta,dotDelta,rl,1/(1+z_ini),deltaT,fscal,ng,local_grid_size.x,local_grid_size.y,local_grid_size.z, nlocal, world_rank);
}