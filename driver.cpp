#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
#include "haccgpm.hpp"
#define GPU
#define alltoall
#include <mpi.h>
#include "swfft-all-to-all/include/swfft.hpp"
#include "cambTools/ccamb.h"
#include "bdwgc/include/gc.h"

int serial(const char* params_file){

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::serial::MemoryManager mem(params);
    HACCGPM::serial::fft_cache_plan(params.ng);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    init_python(0,0);

    if (params.do_analysis){
        import_analysis_module(params.analysis_dir,params.analysis_py);
    }

    HACCGPM::serial::GenerateDisplacementIC(params_file,&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize);
    
    if (!params.do_analysis){
        finalize_python(0);
    }
    
    HACCGPM::serial::InitGreens(mem.d_greens,params.ng,params.blockSize);

    char stepstr[400];
    sprintf(stepstr, "%s.pk.ini", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    sprintf(stepstr, "%s.particles.ini", params.prefix);
    HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);

    ts.advanceHalfStep();

    for (int step = 0; step < params.lastStep; step++){

        printf("\n=========\nSTEP %d\n",step);

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        HACCGPM::serial::CIC(mem.d_grid,mem.d_pos,params.ng,params.blockSize);
        HACCGPM::serial::SolveGradient(mem.d_grad,mem.d_grid,mem.d_greens,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdateVelocities(mem.d_vel,mem.d_grad,mem.d_pos,ts,params.ng,params.blockSize);

        ts.advanceHalfStep();

        HACCGPM::serial::UpdatePositions(mem.d_pos,mem.d_vel,ts,0.5,params.ng,params.blockSize);

        if (params.pks[step]){
            sprintf(stepstr, "%s.pk.%d", params.prefix,step);
            HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);
        }

        if (params.dumps[step]){
            sprintf(stepstr, "%s.particles.%d", params.prefix,step);
            HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);
        }

        if (params.do_analysis){
            if (params.analysis[step]){
                printf("Doing Python Analysis Step\n");
                float* particles = (float*)malloc(sizeof(float)*params.ng*params.ng*params.ng*4);
                cudaCall(cudaMemcpy, particles, mem.d_pos, sizeof(float)*params.ng*params.ng*params.ng*4, cudaMemcpyDeviceToHost);
                call_analysis(step,ts.z,ts.aa,particles,params.ng*params.ng*params.ng,params.ng,params.rl);
                free(particles);
                printf("   Done Python Analysis Step\n");
                cudaCall(cudaDeviceSynchronize);
            }
        }
    }

    sprintf(stepstr, "%s.pk.fin", params.prefix);
    HACCGPM::serial::GetPowerSpectrum(mem.d_pos,mem.d_grid,params.ng,params.rl,221,stepstr,params.pkFolds,params.blockSize);

    sprintf(stepstr, "%s.particles.fin", params.prefix);
    HACCGPM::serial::writeOutput(stepstr,mem.d_pos,mem.d_vel,params.ng);

    if (params.do_analysis){
        finalize_python(0);
    }

    CPUTimer_t end = CPUTimer();

    printf("\n\n=========\nTimers:\n");
    HACCGPM::serial::printCICTimes();
    HACCGPM::serial::printFFTTimes();
    HACCGPM::serial::printPowerTimes();
    HACCGPM::serial::printOutputTimes();
    printf("   Total: %5.2g minutes\n",((double)(end-start)) * 1.66667e-8);
    printf("=========\n\n");

    return 0;
}

int coords2rank(int3 coords, int3 dims){
    return coords.x * dims.y * dims.z + coords.y * dims.z + coords.z;
}

void zero(int* ptr, int n){
    for (int i = 0; i < n; i++){
        ptr[i] = 0;
    }
}

void sends_per_rank(int* n_sends, int* n_swaps, int3 grid_coords, int3 global_grid_size){
    int idx = 0;
    for (int x = -1; x < 2; x++){
        for (int y = -1; y < 2; y++){
            for (int z = -1; z < 2; z++){
                if (!(x == 0 && y == 0 && z == 0)){
                    int3 dest_coords = make_int3((grid_coords.x + x + global_grid_size.x)%global_grid_size.x, 
                                            (grid_coords.y + y + global_grid_size.y)%global_grid_size.y, 
                                            (grid_coords.z + z + global_grid_size.z)%global_grid_size.z);
                    int dest_rank = coords2rank(dest_coords,global_grid_size);
                    n_sends[dest_rank] += n_swaps[idx];
                }
                idx++;
            }
        }
    }
}

void combine_sends(float4** pos_sends, float4** vel_sends, float4** tmp_swap_pos, float4** tmp_swap_vel, int* n_sends, int* n_swaps, int3 grid_coords, int3 global_grid_size, int world_size, int world_rank){
    for (int i = 0; i < world_size; i++){
        if (n_sends[i] > 0){
            pos_sends[i] = (float4*)GC_MALLOC(sizeof(float4)*n_sends[i]);
            vel_sends[i] = (float4*)GC_MALLOC(sizeof(float4)*n_sends[i]);
        }
    }

    int counts[world_size];
    zero(counts,world_size);

    int idx = 0;
    for (int x = -1; x < 2; x++){
        for (int y = -1; y < 2; y++){
            for (int z = -1; z < 2; z++){
                if (!(x == 0 && y == 0 && z == 0)){
                    int3 dest_coords = make_int3((grid_coords.x + x + global_grid_size.x)%global_grid_size.x, 
                                            (grid_coords.y + y + global_grid_size.y)%global_grid_size.y, 
                                            (grid_coords.z + z + global_grid_size.z)%global_grid_size.z);
                    int dest_rank = coords2rank(dest_coords,global_grid_size);
                    for (int i = 0; i < n_swaps[idx]; i++){
                        float4 this_particle = tmp_swap_pos[idx][i];
                        float4 this_vel = tmp_swap_vel[idx][i];
                        int store_idx = counts[dest_rank];
                        counts[dest_rank]++;
                        pos_sends[dest_rank][store_idx] = this_particle;
                        vel_sends[dest_rank][store_idx] = this_vel;
                    }
                }
                idx++;
            }
        }
    }
}

void send_recv_data(int world_size, int world_rank, int* n_sends, int* n_recvs, float4** pos_recvs, float4** vel_recvs, float4** pos_sends, float4** vel_sends){
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Send(&n_sends[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
    }
    for (int i = 0; i < world_size; i++){
        if (i != world_rank)MPI_Recv(&n_recvs[i],1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (n_recvs[i] > 0){
            pos_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
            vel_recvs[i] = (float4*)GC_MALLOC(sizeof(float4)*n_recvs[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            if (n_sends[i] != 0){
                MPI_Send(pos_sends[i],n_sends[i]*4,MPI_FLOAT,i,0,MPI_COMM_WORLD);
                MPI_Send(vel_sends[i],n_sends[i]*4,MPI_FLOAT,i,1,MPI_COMM_WORLD);
            }
        }
    }
    for (int i = 0; i < world_size; i++){
        if (i != world_rank){
            if (n_recvs[i] != 0){
                MPI_Recv(pos_recvs[i],n_recvs[i]*4,MPI_FLOAT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv(vel_recvs[i],n_recvs[i]*4,MPI_FLOAT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void transfer_particles(HACCGPM::Params& params,HACCGPM::parallel::MemoryManager& mem){

    int n_particles = params.frac * params.nlocal;
    int3 local_grid_size = make_int3(params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    int3 global_grid_size = make_int3(params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
    int3 grid_coords = make_int3(params.grid_coords[0],params.grid_coords[1],params.grid_coords[2]);

    //MPI_Barrier(MPI_COMM_WORLD);

    if(params.world_rank == 0)printf("DOING SWAP!!!\n");

    float4* tmp_swap_pos[27];
    float4* tmp_swap_vel[27];

    int n_swaps[27];

    HACCGPM::parallel::loadIntoBuffers(tmp_swap_pos,tmp_swap_vel,n_swaps,mem.d_pos,mem.d_vel,params.nlocal,local_grid_size,n_particles,params.blockSize,params.world_rank);

    int remaining = n_swaps[13];

    int n_recvs[params.world_size];
    zero(n_recvs,params.world_size);
    int n_sends[params.world_size];
    zero(n_sends,params.world_size);
    float4* pos_recvs[params.world_size];
    float4* vel_recvs[params.world_size];
    float4* pos_sends[params.world_size];
    float4* vel_sends[params.world_size];

    sends_per_rank(n_sends,n_swaps,grid_coords,global_grid_size);

    combine_sends(pos_sends,vel_sends,tmp_swap_pos,tmp_swap_vel,n_sends,n_swaps,grid_coords,global_grid_size,params.world_size,params.world_rank);
    
    send_recv_data(params.world_size,params.world_rank,n_sends,n_recvs,pos_recvs,vel_recvs,pos_sends,vel_sends);

    int total_recieved = 0;
    for (int i = 0; i < params.world_size; i++){
        total_recieved += n_recvs[i];
    }

    printf("Rank %d: Total Recieved = %d\n",params.world_rank,total_recieved);

    float4* pos_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);
    float4* vel_to_transfer = (float4*)malloc(sizeof(float4)*total_recieved);
    float4* h_pos = (float4*)malloc(sizeof(float4)*params.n_particles);
    float4* h_vel = (float4*)malloc(sizeof(float4)*params.n_particles);

    cudaCall(cudaMemcpy,h_pos,mem.d_pos,sizeof(float4)*params.n_particles,cudaMemcpyDeviceToHost);
    cudaCall(cudaMemcpy,h_vel,mem.d_vel,sizeof(float4)*params.n_particles,cudaMemcpyDeviceToHost);
    int idx = 0;
    for (int i = 0; i < params.world_size; i++){
        if (i != params.world_rank){
            if (n_recvs[i] != 0){
                for (int j = 0; j < n_recvs[i]; j++){
                    if (idx >= total_recieved){
                        printf("????\n");
                    }
                    pos_to_transfer[idx] = pos_recvs[i][j];
                    vel_to_transfer[idx] = vel_recvs[i][j];
                    idx++;
                }
            }
        }
    }
    idx = remaining;
    for (int i = 0; i < total_recieved; i++){
        float4 tmp_pos = pos_to_transfer[i];
        float4 tmp_vel = vel_to_transfer[i];
        if ((idx) >= params.n_particles){
            printf("PARTICLE OVERFLOW!!!!!!!!!!\n");
        }
        float particle_id = tmp_pos.w;
        bool add = true;
        for (int j = 0; j < params.n_particles; j++){
            if (h_pos[j].w == particle_id){
                add = false;
                printf("Found duplicate particle!!!\n");
                break;
            }
        }
        if (add){
            h_pos[idx] = tmp_pos;
            h_vel[idx] = tmp_vel;
            idx++;
        }
    }

    cudaCall(cudaMemcpy,mem.d_pos,h_pos,sizeof(float4)*params.n_particles,cudaMemcpyHostToDevice);
    cudaCall(cudaMemcpy,mem.d_vel,h_vel,sizeof(float4)*params.n_particles,cudaMemcpyHostToDevice);

    free(h_pos);
    free(h_vel);
    free(pos_to_transfer);
    free(vel_to_transfer);
}

int parallel(const char* params_file){

    GC_INIT();

    cudaFree(0);

    CPUTimer_t start = CPUTimer();

    HACCGPM::Params params = HACCGPM::read_params(params_file);

    HACCGPM::parallel::init_swfft(params);

    if (params.world_rank == 0){
        printf("GLOBAL_GRID_SIZE: [%d %d %d]\n",params.grid_dims[0],params.grid_dims[1],params.grid_dims[2]);
        printf("LOCAL_GRID_SIZE: [%d %d %d]\n",params.local_grid_size[0],params.local_grid_size[1],params.local_grid_size[2]);
    }

    HACCGPM::parallel::MemoryManager mem(params);

    HACCGPM::Timestepper ts(params);
    ts.setInitialZ(params.z_ini);
    ts.reverseHalfStep();

    init_python(0,params.world_rank);

    HACCGPM::parallel::GenerateDisplacementIC(params_file,&mem,params.ng,params.rl,params.z_ini,ts.deltaT,ts.fscal,params.seed,params.blockSize,params.world_rank,params.world_size,params.nlocal,params.local_grid_size);

    transfer_particles(params,mem);

    finalize_python(0);

    HACCGPM::parallel::finalize_swfft();


    return 0;
}

int main(int argc, char** argv){

    MPI_Init(NULL,NULL);

    int world_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    if (argc != 2){
        printf("USAGE: main params\n");
        return 1;
    }
    int out = 1;

    if (world_size == 1){
        printf("\n=========\nRUNNING IN SERIAL MODE\n=========\n");
        out = serial(argv[1]);
    } else{
        if (world_rank == 0){
            printf("\n=========\nRUNNING IN PARLLEL MODE\n=========\n");
        }
        out = parallel(argv[1]);
    }

    MPI_Finalize();

    return out;
}