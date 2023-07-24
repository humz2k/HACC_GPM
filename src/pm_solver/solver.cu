#include <stdlib.h>
#include <stdio.h>
#include "haccgpm.hpp"

#define VerboseSolver

__device__ __forceinline__ float get_gradient(float kmode){
    return sinf(kmode);
}

__global__ void kspace_solve(deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    hostFFT_t greens = __ldg(&d_greens[idx]);
    deviceFFT_t rho = __ldg(&d_rho[idx]);
    rho.x *= greens;
    rho.y *= greens;
    d_rho[idx] = rho;
}

__global__ void kspace_solve_gradient(deviceFFT_t* __restrict d_x, deviceFFT_t* __restrict d_y, deviceFFT_t* __restrict d_z, const deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens, float b1, float b2, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::serial::get_index(idx,ng);
    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    hostFFT_t greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    deviceFFT_t rho = __ldg(&d_rho[idx]);

    deviceFFT_t out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    deviceFFT_t out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    deviceFFT_t out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

__global__ void kspace_solve_gradient_parallel(deviceFFT_t* __restrict d_x, deviceFFT_t* __restrict d_y, deviceFFT_t* __restrict d_z, const deviceFFT_t* __restrict d_rho, const hostFFT_t* __restrict d_greens, float b1, float b2, int ng, int nlocal, int overload, int3 local_grid_size, int3 local_coords){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= nlocal)return;

    double d = ((2*M_PI)/(((double)(ng))));

    int3 idx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);

    float3 kmodes = HACCGPM::get_kmodes(idx3d,ng,d);

    hostFFT_t greens = __ldg(&d_greens[idx]);

    float3 c;
    c.x = -get_gradient(kmodes.x) * greens;
    c.y = -get_gradient(kmodes.y) * greens;
    c.z = -get_gradient(kmodes.z) * greens;

    deviceFFT_t rho = __ldg(&d_rho[idx]);

    deviceFFT_t out_x;
    out_x.x = -c.x * rho.y;
    out_x.y = c.x * rho.x;

    d_x[idx] = out_x;

    deviceFFT_t out_y;
    out_y.x = -c.y * rho.y;
    out_y.y = c.y * rho.x;

    d_y[idx] = out_y;

    deviceFFT_t out_z;
    out_z.x = -c.z * rho.y;
    out_z.y = c.z * rho.x;

    d_z[idx] = out_z;
}

__global__ void combine(float4* __restrict out, const deviceFFT_t* __restrict d_x, const deviceFFT_t* __restrict d_y, const deviceFFT_t* __restrict d_z){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    deviceFFT_t x = __ldg(&d_x[idx]);
    deviceFFT_t y = __ldg(&d_y[idx]);
    deviceFFT_t z = __ldg(&d_z[idx]);

    float4 this_out;
    this_out.x = x.x;
    this_out.y = y.x;
    this_out.z = z.x;

    out[idx] = this_out;
}

__global__ void combine_parallel(float4* __restrict out, const deviceFFT_t* __restrict d_x, const deviceFFT_t* __restrict d_y, const deviceFFT_t* __restrict d_z, int3 local_grid_size, int3 local_coords, int overload, int nlocal, int ng){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nlocal)return;
    deviceFFT_t x = __ldg(&d_x[idx]);
    deviceFFT_t y = __ldg(&d_y[idx]);
    deviceFFT_t z = __ldg(&d_z[idx]);

    int3 globalIdx3d = HACCGPM::parallel::get_global_index(idx,ng,local_grid_size,local_coords);
    int globalIdx = globalIdx3d.x * ng * ng + globalIdx3d.y * ng + globalIdx3d.z;

    float4 this_out;
    this_out.x = x.x;
    this_out.y = y.x;
    this_out.z = z.x;
    this_out.w = globalIdx;

    int3 rhoIdx3d = HACCGPM::parallel::get_local_index(idx,local_grid_size.x,local_grid_size.y,local_grid_size.z);
    rhoIdx3d.x += overload;
    rhoIdx3d.y += overload;
    rhoIdx3d.z += overload;

    int3 overload_dims = make_int3(local_grid_size.x + 2*overload, local_grid_size.y + 2*overload, local_grid_size.z + 2*overload);

    int rhoIdx = rhoIdx3d.x * overload_dims.y * overload_dims.z + rhoIdx3d.y * overload_dims.z + rhoIdx3d.z;

    out[rhoIdx] = this_out;
}

void HACCGPM::parallel::SolveGradient(HACCGPM::Params& params, HACCGPM::parallel::MemoryManager& mem, int calls){
    HACCGPM::parallel::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,mem.d_x,mem.d_y,mem.d_z,params.ng,params.n_particles,params.nlocal,params.local_grid_size_vec,params.grid_coords_vec,params.grid_dims_vec,params.world_rank,params.world_size,params.overload,params.blockSize,calls);
}

void HACCGPM::parallel::SolveGradient(float4* d_grad, deviceFFT_t* d_rho, hostFFT_t* d_greens, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int ng, int n_particles, int nlocal, int3 local_grid_size, int3 local_coords, int3 dims, int world_rank, int world_size, int overload, int blockSize, int calls){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;

    getIndent(calls);

    #ifdef VerboseSolver
    if(world_rank == 0)printf("%sSolveGradent was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    if(world_rank == 0)printf("%s   Doing forward fft...\n",indent);
    #endif

    HACCGPM::parallel::forward_fft(d_rho,ng,calls+1);

    #ifdef VerboseSolver
    if(world_rank == 0)printf("%s      Done forward fft.\n",indent);
    if(world_rank == 0)printf("%s   Calling kspace_solve_gradient_parallel...\n",indent);
    #endif
    double const b1 =   4.0 / 3.0;
    double const b2 = - 1.0 / 6.0;
    InvokeGPUKernelParallel(kspace_solve_gradient_parallel,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,b1,b2,ng,nlocal,overload,local_grid_size,local_coords);
    #ifdef VerboseSolver
    if(world_rank == 0)printf("%s      Called kspace_solve_gradient_parallel.\n",indent);
    if(world_rank == 0)printf("%s   Doing backward ffts...\n",indent);
    #endif

    HACCGPM::parallel::backward_fft(d_x,ng,calls+1);
    HACCGPM::parallel::backward_fft(d_y,ng,calls+1);
    HACCGPM::parallel::backward_fft(d_z,ng,calls+1);

    #ifdef VerboseSolver
    if(world_rank == 0)printf("%s      Done backward ffts.\n",indent);
    if(world_rank == 0)printf("%s   Calling combine_parallel...\n",indent);
    #endif

    InvokeGPUKernelParallel(combine_parallel,numBlocks,blockSize,d_grad,d_x,d_y,d_z,local_grid_size,local_coords,overload,nlocal,ng);

    #ifdef VerboseSolver
    if(world_rank == 0)printf("%s      Done combine_parallel.\n",indent);
    #endif

    HACCGPM::parallel::GridExchange gexch(local_coords,local_grid_size,dims,ng,world_size,world_rank,overload,blockSize);

    gexch.fill(d_grad,calls+1);

    /*int total_grid_size = (local_grid_size.x + overload*2) * (local_grid_size.y + overload*2) * (local_grid_size.z + overload*2);

    float4* h_grad = (float4*)malloc(sizeof(float4)*total_grid_size);
    cudaCall(cudaMemcpy,h_grad,d_grad,sizeof(float4)*total_grid_size,cudaMemcpyDeviceToHost);

    if (world_rank == 0){
        //int i = 0;
        for (int x = -overload; x < 0; x++){
            for (int y = -overload; y < 0; y++){
                for (int z = -overload; z < 0; z++){
                    int3 global_idx3d = make_int3(x + local_coords.x * local_grid_size.x,y + local_coords.y * local_grid_size.y, z + local_coords.z * local_grid_size.z);
                    int global_idx = ((global_idx3d.x + ng)%ng) * ng * ng + ((global_idx3d.y + ng)%ng) * ng + ((global_idx3d.z+ng)%ng);
                    int i = (x + overload) * (local_grid_size.y + overload*2) * (local_grid_size.z + overload*2) + (y + overload) * (local_grid_size.z + overload*2) + (z + overload);
                    printf("%d = (%d %d %d) = %g %g %g %g\n",global_idx,x,y,z,h_grad[i].x,h_grad[i].y,h_grad[i].z,h_grad[i].w);
                    //i++;
                    //if (i > 10)break;
                }
                //if (i > 10)break;
            }
            //if (i > 10)break;
        }
    }

    free(h_grad);*/
}

void HACCGPM::serial::Solve(deviceFFT_t* d_rho, hostFFT_t* d_greens, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseSolver
    printf("%sSolve was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Doing forward fft...\n",indent);
    #endif
    HACCGPM::serial::forward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done forward fft.\n",indent);
    printf("%s   Calling kspace_solve...\n",indent);
    #endif
    InvokeGPUKernel(kspace_solve,numBlocks,blockSize,d_rho,d_greens);
    #ifdef VerboseSolver
    printf("%s      Called kspace_solve.\n",indent);
    printf("%s   Doing backward fft...\n",indent);
    #endif
    HACCGPM::serial::backward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done backward fft.\n",indent);
    #endif
}

void HACCGPM::serial::SolveGradient(HACCGPM::Params& params, HACCGPM::serial::MemoryManager& mem, int calls){
    HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,mem.d_x,mem.d_y,mem.d_z,params.ng,params.blockSize,calls);
}

void HACCGPM::serial::SolveGradient(float4* d_grad, deviceFFT_t* d_rho, hostFFT_t* d_greens, deviceFFT_t* d_x, deviceFFT_t* d_y, deviceFFT_t* d_z, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng)/blockSize;

    getIndent(calls);

    #ifdef VerboseSolver
    printf("%sSolveGradient was called with\n%s   blockSize %d\n%s   numBlocks %d\n",indent,indent,blockSize,indent,numBlocks);
    printf("%s   Doing forward fft...\n",indent);
    #endif
    HACCGPM::serial::forward_fft(d_rho,ng,calls+1);
    #ifdef VerboseSolver
    printf("%s      Done forward fft.\n",indent);
    printf("%s   Calling kspace_solve_gradient...\n",indent);
    #endif

    double const b1 =   4.0 / 3.0;
    double const b2 = - 1.0 / 6.0;

    InvokeGPUKernel(kspace_solve_gradient,numBlocks,blockSize,d_x,d_y,d_z,d_rho,d_greens,b1,b2,ng);

    #ifdef VerboseSolver
    printf("%s      Called kspace_solve_gradient.\n",indent);
    printf("%s   Doing backward ffts...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(d_x,ng,calls+1);
    HACCGPM::serial::backward_fft(d_y,ng,calls+1);
    HACCGPM::serial::backward_fft(d_z,ng,calls+1);

    //SHOULD WE NORMALIZE THIS???

    #ifdef VerboseSolver
    printf("%s      Done backward ffts.\n",indent);
    printf("%s   Calling combine...\n",indent);
    #endif

    InvokeGPUKernel(combine,numBlocks,blockSize,d_grad,d_x,d_y,d_z);

    #ifdef VerboseSolver
    printf("%s      Called combine.\n",indent);
    #endif
}