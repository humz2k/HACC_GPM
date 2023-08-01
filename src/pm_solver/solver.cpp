#include <stdlib.h>
#include <stdio.h>
#include "kernels.hpp"

//#define VerboseSolver

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

    launch_kspace_solve_gradient(d_x,d_y,d_z,d_rho,d_greens,ng,nlocal,overload,local_grid_size,local_coords,world_rank,numBlocks,blockSize,calls);

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

    launch_combine(d_grad,d_x,d_y,d_z,local_grid_size,local_coords,overload,nlocal,ng,world_rank,numBlocks,blockSize,calls);

    #ifdef VerboseSolver
    if(world_rank == 0)printf("%s      Done combine_parallel.\n",indent);
    #endif

    HACCGPM::parallel::GridExchange gexch(local_coords,local_grid_size,dims,ng,world_size,world_rank,overload,blockSize);

    gexch.fill(d_grad,calls+1);

    
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

    launch_kspace_solve(d_rho,d_greens,numBlocks,blockSize,calls);

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
    #ifdef USE_GREENS_CACHE
        #ifdef USE_ONE_GRID
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,params.ng,params.blockSize,calls);
        #else
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,mem.d_x,mem.d_y,mem.d_z,params.ng,params.blockSize,calls);
        #endif
    #else
        #ifdef USE_ONE_GRID
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,params.ng,params.blockSize,calls);
        #else
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_x,mem.d_y,mem.d_z,params.ng,params.blockSize,calls);
        #endif
    #endif
}

template<class T1, class T2>
void HACCGPM::serial::SolveGradient(float4* d_grad, T1* d_rho, T2* d_greens, T1* d_x, T1* d_y, T1* d_z, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng + (blockSize - 1))/blockSize;

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

    launch_kspace_solve_gradient(d_x,d_y,d_z,d_rho,d_greens,ng,numBlocks,blockSize,calls);

    #ifdef VerboseSolver
    printf("%s      Called kspace_solve_gradient.\n",indent);
    printf("%s   Doing backward ffts...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(d_x,ng,calls+1);
    HACCGPM::serial::backward_fft(d_y,ng,calls+1);
    HACCGPM::serial::backward_fft(d_z,ng,calls+1);

    #ifdef VerboseSolver
    printf("%s      Done backward ffts.\n",indent);
    printf("%s   Calling combine...\n",indent);
    #endif

    launch_combine(d_grad,d_x,d_y,d_z,ng,numBlocks,blockSize,calls);

    #ifdef VerboseSolver
    printf("%s      Called combine.\n",indent);
    #endif
}

template void HACCGPM::serial::SolveGradient<deviceFFT_t,hostFFT_t>(float4*,deviceFFT_t*,hostFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<deviceFFT_t,float>(float4*,deviceFFT_t*,float*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t,hostFFT_t>(float4*,floatFFT_t*,hostFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t,float>(float4*,floatFFT_t*,float*,floatFFT_t*,floatFFT_t*,floatFFT_t*,int,int,int);

template<class T>
void HACCGPM::serial::SolveGradient(float4* d_grad, T* d_rho, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng + (blockSize - 1))/blockSize;

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

    launch_grid2float4(d_grad,d_rho,ng,numBlocks,blockSize,calls);

    for (int i = 0; i < 3; i++){
        launch_kspace_solve_gradient(d_rho,d_grad,i,ng,numBlocks,blockSize,calls);

        HACCGPM::serial::backward_fft(d_rho,ng,calls+1);

        launch_get_real_grid(d_rho,d_grad,i,ng,numBlocks,blockSize,calls);
    }

}

template void HACCGPM::serial::SolveGradient<deviceFFT_t>(float4*,deviceFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t>(float4*,floatFFT_t*,int,int,int);

template<class T1, class T2>
void HACCGPM::serial::SolveGradient(float4* d_grad, T1* d_rho, T2* d_greens, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng + (blockSize - 1))/blockSize;

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

    launch_grid2float4(d_grad,d_rho,ng,numBlocks,blockSize,calls);

    for (int i = 0; i < 3; i++){
        launch_kspace_solve_gradient(d_rho,d_grad,d_greens,i,ng,numBlocks,blockSize,calls);

        HACCGPM::serial::backward_fft(d_rho,ng,calls+1);

        launch_get_real_grid(d_rho,d_grad,i,ng,numBlocks,blockSize,calls);
    }
}

template void HACCGPM::serial::SolveGradient<deviceFFT_t,hostFFT_t>(float4*,deviceFFT_t*,hostFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<deviceFFT_t,float>(float4*,deviceFFT_t*,float*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t,hostFFT_t>(float4*,floatFFT_t*,hostFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t,float>(float4*,floatFFT_t*,float*,int,int,int);

template<class T>
void HACCGPM::serial::SolveGradient(float4* d_grad, T* d_rho, T* d_x, T* d_y, T* d_z, int ng, int blockSize, int calls){
    int numBlocks = (ng*ng*ng + (blockSize - 1))/blockSize;

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

    launch_kspace_solve_gradient(d_x,d_y,d_z,d_rho,ng,numBlocks,blockSize,calls);

    #ifdef VerboseSolver
    printf("%s      Called kspace_solve_gradient.\n",indent);
    printf("%s   Doing backward ffts...\n",indent);
    #endif

    HACCGPM::serial::backward_fft(d_x,ng,calls+1);
    HACCGPM::serial::backward_fft(d_y,ng,calls+1);
    HACCGPM::serial::backward_fft(d_z,ng,calls+1);

    #ifdef VerboseSolver
    printf("%s      Done backward ffts.\n",indent);
    printf("%s   Calling combine...\n",indent);
    #endif

    launch_combine(d_grad,d_x,d_y,d_z,ng,numBlocks,blockSize,calls);

    #ifdef VerboseSolver
    printf("%s      Called combine.\n",indent);
    #endif
}

template void HACCGPM::serial::SolveGradient<deviceFFT_t>(float4*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,deviceFFT_t*,int,int,int);
template void HACCGPM::serial::SolveGradient<floatFFT_t>(float4*,floatFFT_t*,floatFFT_t*,floatFFT_t*,floatFFT_t*,int,int,int);